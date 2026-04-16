# path: apps/aquaculture_ur10_split_scan_phased_follow.py
# Isaac Sim 5.0.0  (omni.isaac.core)
# - Phased sequence with transit height and target "follow" (gated + linear interpolation)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import math
import numpy as np
from typing import Optional, Tuple, List

from pxr import UsdGeom, Gf, UsdLux, UsdPhysics, Sdf
from omni.isaac.core import World
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.robots import Robot
from omni.isaac.core.controllers import ArticulationController

#==============#============================================
from omni.isaac.core.objects import DynamicCuboid
from pxr import UsdShade, Sdf, Gf, Tf, UsdGeom
from isaacsim.sensors.physics import ContactSensor
#==============#============================================

# ========================== CONFIG ==========================
GRID_SIZE            = 4
TANK_SIZE            = 1.2
TANK_HEIGHT          = 0.8
WALL_THICK           = 0.05
SPACING              = 0.4

RAIL_HEIGHT          = 0.8
CARRIAGE_Z           = RAIL_HEIGHT + 0.125
TRANSIT_Z            = CARRIAGE_Z + 0.25  # ⭐ 탱크 간 이동 시 캐리지 상승 높이

# IK/FD
EE_OFFSET            = np.array([0.0, 0.0, 0.10])
FD_H                 = 1e-3
FD_STEPS_PER_COL     = 1
KP                   = 6.0
LAMBDA               = 0.03
DQ_LIMIT             = 0.8
POS_TOL              = 0.012
MAX_IK_ITERS         = 40

# Safety / Reach
WALL_MARGIN          = 0.06
EDGE_CLEAR           = 0.015
ARM_REACH_X          = 0.65
Z_FLOOR_CLEAR        = 0.02

#ori
ORI_WEIGHT          = 1.0   # 자세 에러 가중치 (0.5~2.0 사이 튜닝)
ORI_TOL             = 3.0 * math.pi / 180.0   # 3도 이하면 OK

# Raster / dwells
STEP_Y               = 0.15
STEP_X               = 0.15
LINE_DWELL           = 0.4
POINT_DWELL          = 0.05

# Target follow (gated + linear)
TARGET_SIZE          = 0.01
FOLLOW_START_DIST    = 0.12   # ⭐ EE-타깃 거리가 이 이하면 타깃 이동 시작
FOLLOW_HOLD_DIST     = 0.15   # ⭐ 이보다 멀어지면 타깃 정지(EE 기다림)
TARGET_SPEED         = 0.4   # ⭐ 타깃 선형 속도 [m/s]
TARGET_MAX_STEP      = 0.04   # 안전용: 프레임당 최대 이동 [m]

# Preset joint poses (deg→rad); 필요에 맞게 조정
READY_POSE           = np.deg2rad(np.array([  0,   0,  90,  -90,  -90,   0], dtype=np.float64))
READY_POSE2          = np.deg2rad(np.array([180,   0,  90,  -90,  -90,   0], dtype=np.float64))
TRANSIT_POSE         = np.deg2rad(np.array([  0, -90,  90,  -90,  -90,   0], dtype=np.float64))
TRANSIT_POSE2        = np.deg2rad(np.array([180, -90,  90,  -90,  -90,   0], dtype=np.float64))
POSE_STEPS           = 120
# ===========================================================

class AquacultureGantrySystem:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.stage = get_current_stage()

        self.grid_size = GRID_SIZE
        self.tank_size = TANK_SIZE
        self.tank_height = TANK_HEIGHT
        self.wall_thickness = WALL_THICK
        self.spacing = SPACING
        self.rail_height = RAIL_HEIGHT

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = CARRIAGE_Z

        self.tank_positions = {}  # (row,col)->(x,y)

        self.num_robots = 2
        self.robot_prim_paths: List[str] = []
        self.robots: List[Robot] = []
        self.ee_prims: List[XFormPrim] = []
        self.carriage_paths: List[str] = []
        self.master_robot_id: int = 0  # 0번 로봇만 IK 계산, 나머지는 복제
        self.row_offsets = []   # Y축 오프셋
        for i in range(self.num_robots):
            row_y = self._tank_center(i, 0)[1]   # row i 의 중심 Y값
            self.row_offsets.append(row_y)
        self.current_y = self.row_offsets[self.master_robot_id]
        self.robot: Optional[Robot] = None
        self.ee_prim: Optional[XFormPrim] = None

        self.target: Optional[VisualCuboid] = None
        self._J_cache = {"J": None, "q": None, "p": None}  # ✅ Broyden용 캐시
#==============#============================================

        self._water_timer = 0.0
        self._water_hidden = False
        self.water_prims = []  # 생성된 물 프림 경로 모음
        self.item_contacts: list[dict] = []   # {'obj': DynamicCuboid, 'sensor': ContactSensor, 'prim_path': str}
        self.debug_forces = []          # 마지막 프레임의 force 값들만 저장

        self.ur10_articulation = None
        self.has_ur10 = False
  
        
        # 갠트리 시스템 파라미터
        self.rail_height = 0.8  # 레일 높이 (m)
        self.rail_width = 0.4

        # __init__ 끝부분에 추가

    def enable_collision(self, prim_path: str, mode: str = "static", mass: float = 1.0):
        """
        USD 프림에 물리/충돌 스키마 적용.
        mode: "static" | "dynamic"
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim not found: {prim_path}")

        # 충돌 활성
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)

        if mode == "dynamic":
            # 동적 강체 설정
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass) if mass > 0 else 1.0)
        elif mode == "static":
            # 정적은 RigidBody 불필요
            pass
        else:
            raise ValueError("mode must be 'static' or 'dynamic'")
        
    def make_metal(self, prim_path, color=(0.7, 0.7, 0.75), metallic=1.0, roughness=0.6):
        stage = self.stage
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return  # 없는 경로면 스킵

        mat = UsdShade.Material.Define(stage, prim_path + "/__Metal")
        sh  = UsdShade.Shader.Define(stage, prim_path + "/__Metal/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(float(metallic))
        sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(float(roughness))
        mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        UsdShade.MaterialBindingAPI(prim).Bind(mat)
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(prim).CreateDoubleSidedAttr(False)

    def make_plastic_yellow(self, prim_path,
                        base_color=(0.98, 0.86, 0.12),  # 노란 플라스틱 톤
                        roughness=0.35,                 # 약간 번들거림
                        specular=0.5,                   # 플라스틱 하이라이트
                        ior=1.46):                      # 플라스틱 IOR
        stage = self.stage
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return

        mat_path = prim_path + "/__PlasticYellow"
        mat = UsdShade.Material.Define(stage, mat_path)
        sh  = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")

        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*base_color))
        sh.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)             # 금속성 X
        sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(float(roughness))
        sh.CreateInput("specular",     Sdf.ValueTypeNames.Float).Set(float(specular))
        sh.CreateInput("ior",          Sdf.ValueTypeNames.Float).Set(float(ior))
        # 필요시 투명 플라스틱 느낌: sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.8)

        # 연결 & 바인딩
        mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        UsdShade.MaterialBindingAPI(prim).Bind(mat)
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(prim).CreateDoubleSidedAttr(True)

    def material_change(self):

        for p in (
            #[f"/World/Gantry/FixedRails/XRail_{i}" for i in range(self.grid_size+1)] +
            #[f"/World/Gantry/FixedRails/YRail_{i}" for i in range(self.grid_size+1)] +
            [f"/World/Gantry/Support_{i}" for i in range(4)] +
            ["/World/Gantry/YBeam"]#, "/World/Gantry/RobotCarriage"]
        ):
            self.make_metal(p)  # <-- stage 넘기지 마세요
         
        for q in (
            [f"/World/Gantry/FixedRails/XRail_{i}" for i in range(self.grid_size+1)] + \
            [f"/World/Gantry/FixedRails/YRail_{i}" for i in range(self.grid_size+1)]
        ):
            self.make_plastic_yellow(q, roughness=0.45)  # 레일은 살짝 더 무광

    def ensure_no_physics(self, prim_path: str) -> None:
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim not found: {prim_path}")
        attr = prim.GetAttribute("physics:collisionEnabled")
        if not attr.IsValid():
            attr = prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool)
        attr.Set(False)

    def bind_transparent_material(
        self,
        prim_path: str,
        *,
        base_color=(0.15, 0.45, 1.0),
        opacity=0.08,
        roughness=0.02,
        ior=1.33,
        double_sided=True,
    ) -> None:
        stage = self.stage
        mat_path = prim_path + "_Material"

        # 1) Material & Shader 생성
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path + "/PBRShader")
        shader.CreateIdAttr("UsdPreviewSurface")

        # 2) Shader 입력 세팅
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*base_color))
        shader.CreateInput("opacity",      Sdf.ValueTypeNames.Float).Set(float(opacity))
        shader.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("ior",          Sdf.ValueTypeNames.Float).Set(float(ior))

        # 3) Shader의 surface Output을 명시적으로 만들고 Material의 Surface Output과 연결
        shader_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mtlsurf_out = material.CreateSurfaceOutput()  # (렌더 컨텍스트 기본값)
        mtlsurf_out.ConnectToSource(shader_out)       # <-- 핵심 수정

        # 4) 바인딩 + 양면
        target_prim = stage.GetPrimAtPath(prim_path)
        UsdShade.MaterialBindingAPI(target_prim).Bind(material)
        # Cube도 Gprim이므로 doubleSided 적용 가능
        UsdGeom.Gprim(target_prim).CreateDoubleSidedAttr(bool(double_sided))

    def create_water_volume(self, row: int, col: int) -> None:
        """수조 내부를 채우는 투명/비충돌 '물' 큐브 생성."""
        cx, cy = self.tank_positions[(row, col)]
        inner_xy = self.tank_size - 2.0 * self.wall_thickness
        water_h = self.tank_height * 0.8                # 원하는 수면 높이 비율
        z_bottom = self.wall_thickness                  # 바닥 위
        z_center = z_bottom + water_h / 2.0

        prim_path = f"/World/Tanks/Tank_{row}_{col}/Water"
        # USD 큐브(충돌 지정하지 않음!)
        cube = UsdGeom.Cube.Define(self.stage, prim_path)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(cx, cy, z_center))
        xform.AddScaleOp().Set(Gf.Vec3f(inner_xy, inner_xy, water_h))

        # 물리/충돌 제거 + 투명 머티리얼 바인딩
        self.ensure_no_physics(prim_path)
        self.bind_transparent_material(
            prim_path,
            base_color=(0.15, 0.45, 1.0),
            opacity=0.25,     # 더 투명하게 하려면 낮추기(예: 0.15)
            roughness=0.02,
            ior=1.33
        )
        self.water_prims.append(prim_path)

    def create_poos_3(
        self,
        row: int,
        col: int,
        count: int = 10,
        cube_scale=np.array([0.2, 0.2, 0.2], dtype=float),
        margin: float = 0.05,
        min_gap: float = 0.10,
        z_clearance: float = 0.02,
        max_tries_per_item: int = 60,
        spawn_impactors: bool = False,   # 사용 안 함(호환용)
        drop_offset_z: float = 1.0,
        drop_scale=np.array([0.10, 0.10, 0.10], dtype=float),
        drop_mass: float = 5.0
    ):
        """
        각 탱크 안에 동적 큐브(찌꺼기) + 접촉 센서를 생성.
        - omni/isaac 두 구현의 인자 차이(size vs scale)를 try/except로 자동 호환.
        """
        # 탱크 중심과 내부 가용 영역 계산
        if (row, col) not in self.tank_positions:
            raise RuntimeError(f"Tank center not found for ({row}, {col}). Make sure _tank() ran before create_poos_3().")

        cx, cy = self.tank_positions[(row, col)]
        inner = float(self.tank_size - 2.0 * self.wall_thickness - 2.0 * margin)
        if inner <= 0.1:
            inner = 0.1

        half = inner / 2.0
        z_start = float(self.wall_thickness) + float(z_clearance)

        # 부모 Xform (Items) 보장
        items_root = f"/World/Tanks/Tank_{row}_{col}/Items"
        if not self.stage.GetPrimAtPath(items_root):
            UsdGeom.Xform.Define(self.stage, items_root)

        # 포인트 샘플링(간격 제약)
        placed = []
        rng = np.random.default_rng()

        def valid(p):
            for q in placed:
                if (p[0] - q[0])**2 + (p[1] - q[1])**2 < (min_gap ** 2):
                    return False
            return True

        for _ in range(count):
            ok = False
            for _try in range(max_tries_per_item):
                x = cx + rng.uniform(-half, half)
                y = cy + rng.uniform(-half, half)
                if valid((x, y)):
                    placed.append((x, y))
                    ok = True
                    break
            if not ok:
                break  # 밀도 과다 시 조기 종료

        # 생성 루프
        for idx, (px, py) in enumerate(placed):
            prim_path = f"{items_root}/item_{idx}"
            name = f"tank{row}_{col}_item{idx}"

            # ✅ size는 스칼라, 실제 크기는 scale로 지정
            cube = DynamicCuboid(
                prim_path=prim_path,
                name=name,
                position=np.array([px, py, z_start], dtype=float),
                size=float(2.0),                                  # <- 스칼라만!
                scale=np.array(cube_scale, dtype=float),          # <- 실제 크기
                color=np.array([0.55, 0.27, 0.07], dtype=float),
            )

            try:
                self.world.scene.add(cube)
            except Exception:
                pass

            sensor = ContactSensor(
                prim_path=prim_path + "/ContactSensor",
                name=f"cs_t{row}_{col}_{idx}",
                frequency=5,
                min_threshold=0.0,
                max_threshold=10_000_000.0,
                radius=float(max(cube_scale)) * 1.8,
            )

            self.item_contacts.append({
                "obj": self.world.scene.get_object(name) if hasattr(self.world.scene, "get_object") else None,
                "sensor": sensor,
                "prim_path": prim_path,
                "name": name,
            })
            # print('pooops made!')

        # # 초기 프레임 몇 번 돌려서 시각화/물리 안정화
        # for _ in range(2):
        #     self.world.step(render=True)



    def setup_camera(self):
        """
        수조 전체의 정중앙 상공에 카메라 1대를 배치(탑뷰).
        높이는 기존 탑뷰 기준(self.tank_height*10)의 2배로 설정.
        """
        # 수조 중심들 평균으로 전체 중앙 계산
        if not self.tank_positions:
            cx, cy = 0.0, 0.0
        else:
            xs, ys = zip(*self.tank_positions.values())
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)

        # 탑뷰 높이(기존 10*h의 두 배)
        z_cam = max(self.tank_height * 10.0, 2.0) * 1.8  # == self.tank_height * 20.0 (최소 4.0 보장)

        cam_path = "/World/Camera/TopView"
        cam = UsdGeom.Camera.Define(self.stage, cam_path)

        # 위에서 아래(-Z)로 내려다보는 탑뷰: 회전 불필요, 위치만 지정
        xform = UsdGeom.Xformable(cam)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(cx, cy, z_cam))

        # (선택) 조금 넓게 보이도록 초점거리 낮춤
        try:
            cam.CreateFocalLengthAttr(18.0)
        except Exception:
            pass

        print(f"Top-view camera placed at ({cx:.3f}, {cy:.3f}, {z_cam:.3f})")

    def setup_row_cameras_on_gantry(self,
                                parent_path="/World/Gantry/YBeam",
                                group_name="RowCams",
                                z_offset_down= 1.0,     # 빔 아래로 내리는 거리
                                focal_length=6.0,      # 살짝 와이드
                                clipping=(0.01, 1000.0)):
        """
        갠트리 YBeam 아래에 카메라 4대 설치(각 row를 수직 하향으로 촬영).
        - 카메라는 parent_path의 자식으로 붙어서 X 이동을 함께 따라감
        - 각 카메라는 해당 row 중심의 Y 위치에 배치
        - 모두 -Z 방향(탑뷰)으로 바라봄 (기본 카메라 방향이 -Z)
        """
        # 부모 확인
        parent = self.stage.GetPrimAtPath(parent_path)
        if not parent or not parent.IsValid():
            raise RuntimeError(f"Parent prim not found: {parent_path}")

        # 카메라 그룹 Xform 생성
        cams_root = f"{parent_path}/{group_name}"
        if not self.stage.GetPrimAtPath(cams_root):
            UsdGeom.Xform.Define(self.stage, cams_root)

        # YBeam의 현재 월드 좌표 가져오기 (배치 기준 z 계산)
        # 여기서는 설계값을 그대로 사용
        z_beam = self.rail_height + 0.1
        z_cam  = z_beam - float(z_offset_down)

        # 각 row의 중심 y 위치를 계산해서 4대 카메라 배치
        for r in range(self.grid_size):
            if (r, 0) in self.tank_positions:
                _, y_row = self.tank_positions[(r, 0)]
            else:
                _, y_row = self._tank_center(r, 0)

            cam_path = f"{cams_root}/Row{r+1}"
            cam = UsdGeom.Camera.Define(self.stage, cam_path)

            xform = UsdGeom.Xformable(cam)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(0.0, float(y_row), float(z_cam)))
            # -Z(아래) 바라봄: 회전 불필요

            try:
                cam.CreateFocalLengthAttr(float(focal_length))  # 🔧 8mm
                cam.CreateClippingRangeAttr(Gf.Vec2f(float(clipping[0]), float(clipping[1])))
            except Exception:
                pass

            try:
                cam.CreatePrimvar("displayName", Sdf.ValueTypeNames.String, Sdf.VariabilityUniform).Set(f"Row{r+1}Cam")
            except Exception:
                pass

        print(f"Installed {self.grid_size} row cameras under {parent_path}.")

    




    def item_contact_and_hide_water(self):
        def item_contact_watch(dt: float):
            if not self.item_contacts:
                self.world.remove_physics_callback("item_contact_watch")
                return

            kept = []
            forces = []            # 이번 스텝의 모든 force를 수집
            name_forces = []       # (이름, force) 목록

            for entry in self.item_contacts:
                obj = entry["obj"]
                sensor = entry["sensor"]
                name = entry.get("name", entry.get("prim_path", "unknown"))

                try:
                    data = sensor.get_current_frame()
                except Exception:
                    kept.append(entry)
                    continue

                # 버전별 키 차이를 흡수
                force = None
                if isinstance(data, dict):
                    for k in ("force", "force_magnitude", "total_force", "norm_force"):
                        if k in data:
                            try:
                                force = float(data[k])
                            except Exception:
                                pass
                            break

                if force is not None:
                    forces.append(force)
                    name_forces.append((name, force))

                # 임계값 체크(개별 제거 로직 유지)
                if force is not None and force > 0.6:
                    obj.set_visibility(visible=False)
                    obj.set_collision_enabled(False)
                    # 제거된 항목은 kept에 넣지 않음
                else:
                    kept.append(entry)

            self.item_contacts = kept
            self.debug_forces = forces  # 외부에서 최근 프레임 force 배열 확인 가능

            # ===== 출력(프레임마다 요약) =====
            # 너무 자주 찍히면 느릴 수 있으니 10프레임마다 찍고 싶으면 아래 두 줄 주석 해제
            # self._contact_debug_tick += 1
            # if self._contact_debug_tick % 10 != 0: return

            # 각 찌꺼기 큐브들 센서 잘받아오는지 확인코드 실행
            if forces:
                fmax = max(forces)
                fmean = sum(forces) / len(forces)
                # print(f"[contact] {len(forces)} readings | max={fmax:.3f}, mean={fmean:.3f}")
                # 상세 목록(원하면 주석 해제)
                # print("  " + ", ".join(f"{n}:{v:.3f}" for n, v in name_forces))
            else:
                print("[contact] no readings this frame")

        
        self.world.add_physics_callback("item_contact_watch", item_contact_watch)

        def _hide_water_after_5s(dt: float):
            if self._water_hidden:
                return
            self._water_timer += float(dt)
            if self._water_timer >= 10:
                # why: 렌더에서 완전히 숨김(투명도 X, 진짜 invisible)
                for p in self.water_prims:
                    prim = self.stage.GetPrimAtPath(p)
                    if prim and prim.IsValid():
                        UsdGeom.Imageable(prim).MakeInvisible()
                self._water_hidden = True
                self.world.remove_physics_callback("hide_water_after_5s")

        self.world.add_physics_callback("hide_water_after_5s", _hide_water_after_5s)

    def create_all_tanks(self):
        """4x4 수조 배열 생성"""
        print("Creating 4x4 tank array...")
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self._tank(row, col)
                self.create_poos_3(row, col, count=3, cube_scale=np.array([0.08, 0.08, 0.08]), min_gap=0.10)

    def create_two_horizontal_beams(self):
        """
        화면에 보이기만 하는 가로빔 2개(위/아래) 생성.
        - 충돌/물리 OFF
        - 길이: 수조 배열 전체 폭
        - 위치: y=±total/2, z=self.rail_height+0.1
        """
        total = self.grid_size * (self.tank_size + self.spacing)
        z = self.rail_height + 0.1
        beam_thick_y = 0.2   # 빔 자체의 '폭'(Y)
        beam_thick_z = 0.1   # 빔 두께(Z)

        for name, y in (("Top",  total/2.0), ("Bottom", -total/2.0)):
            path = f"/World/Gantry/Side{name}"
            cube = UsdGeom.Cube.Define(self.stage, path)
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, y, z))
            xf.AddScaleOp().Set(Gf.Vec3f(total, beam_thick_y, beam_thick_z))
            # 보기용: 물리/충돌 끄기
            self.ensure_no_physics(path)
            # (선택) 금속 재질
            try:
                self.make_metal(path)
            except Exception:
                pass
#============#=============================================================

        # 🔹 EE 목표 자세(회전) 저장용
        self.ori_target_R: Optional[np.ndarray] = None

    # ---------- geom ----------
    def _cube(self, prim_path, position, size, orientation=None):
        cube = UsdGeom.Cube.Define(self.stage, prim_path)
        xform = UsdGeom.XformCommonAPI(cube)
        xform.SetTranslate(Gf.Vec3d(*position))
        xform.SetRotate(Gf.Vec3f(*(orientation or (0.0, 0.0, 0.0))))
        cube.CreateSizeAttr(1.0)
        UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        self.enable_collision(prim_path, 'static', mass=1.0)
        return cube

    def _ground(self):
        g = self.grid_size * (self.tank_size + self.spacing) + 2.0
        self._cube("/World/Ground", [0, 0, -0.05], [g, g, 0.1])

    def _tank_center(self, row, col):
        x = (col - self.grid_size/2 + 0.5) * (self.tank_size + self.spacing)
        y = (row - self.grid_size/2 + 0.5) * (self.tank_size + self.spacing)
        return x, y

    def _tank_group(self, row, col):
        return f"/World/Tanks/Tank_{row}_{col}"
    
    def ensure_physics_scene(self) -> str:
        """스테이지에 물리 씬이 없으면 생성하고 경로 반환."""
        scene_path = "/World/physicsScene"
        if not self.stage.GetPrimAtPath(scene_path):
            UsdPhysics.Scene.Define(self.stage, scene_path)
        return scene_path

    def enable_collision(self, prim_path: str, mode: str = "static", mass: float = 1.0):
        """
        USD 프림에 물리/충돌 스키마 적용.
        mode: "static" | "dynamic"
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim not found: {prim_path}")

        # 충돌 활성
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)

        if mode == "dynamic":
            # 동적 강체 설정
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass) if mass > 0 else 1.0)
        elif mode == "static":
            # 정적은 RigidBody 불필요
            pass
        else:
            raise ValueError("mode must be 'static' or 'dynamic'")


    def _tank(self, row, col):
        x, y = self._tank_center(row, col)
        self.tank_positions[(row, col)] = (x, y)
        z_mid = self.tank_height / 2.0
        self._cube(f"{self._tank_group(row,col)}/Floor",
                   [x, y, self.wall_thickness/2.0],
                   [self.tank_size, self.tank_size, self.wall_thickness])
        walls = [
            [ self.tank_size/2 + self.wall_thickness/2, 0,  self.wall_thickness, self.tank_size, "E"],
            [-self.tank_size/2 - self.wall_thickness/2, 0,  self.wall_thickness, self.tank_size, "W"],
            [0,  self.tank_size/2 + self.wall_thickness/2,  self.tank_size, self.wall_thickness, "N"],
            [0, -self.tank_size/2 - self.wall_thickness/2,  self.tank_size, self.wall_thickness, "S"],
        ]
        for dx, dy, sx, sy, tag in walls:
            self._cube(f"{self._tank_group(row,col)}/Wall_{tag}",
                       [x + dx, y + dy, z_mid], [sx, sy, self.tank_height])
#============#================================
        self.create_water_volume(row, col)
#============#================================
    # def _rails(self):
    #     total = self.grid_size * (self.tank_size + self.spacing)
    #     for i in range(self.grid_size + 1):
    #         y = -total/2 + i * (self.tank_size + self.spacing)
    #         self._cube(f"/World/Gantry/FixedRails/XRail_{i}", [0, y, self.rail_height], [total, 0.4, 0.1])
    #     for i in range(self.grid_size + 1):
    #         x = -total/2 + i * (self.tank_size + self.spacing)
    #         self._cube(f"/World/Gantry/FixedRails/YRail_{i}", [x, 0, self.rail_height], [0.4, total, 0.1])
    #     self._cube("/World/Gantry/YBeam", [0, 0, self.rail_height + 0.1], [0.2, total, 0.05])

    #     # Carriage group only (UR10 as child)
    #     UsdGeom.Xform.Define(self.stage, "/World/Gantry/RobotCarriage")
    # =========================================================================
    def _rails(self):
        total = self.grid_size * (self.tank_size + self.spacing)

        # 고정 레일(기존 그대로)
        for i in range(self.grid_size + 1):
            y = -total/2 + i * (self.tank_size + self.spacing)
            self._cube(f"/World/Gantry/FixedRails/XRail_{i}",
                       [0, y, self.rail_height], [total, 0.4, 0.1])
        for i in range(self.grid_size + 1):
            x = -total/2 + i * (self.tank_size + self.spacing)
            self._cube(f"/World/Gantry/FixedRails/YRail_{i}",
                       [x, 0, self.rail_height], [0.4, total, 0.1])

        # Y 방향 빔 (시각적인 것)
        self._cube("/World/Gantry/YBeam",
                   [0, 0, self.rail_height + 0.1], [0.2, total, 0.05])

        # 🔹 Carriage 그룹: 로봇당 하나씩
        #   /World/Gantry/RobotCarriage_0
        #   /World/Gantry/RobotCarriage_1
        #   ...
        self.carriage_paths = []
        for i in range(self.num_robots):
            c_path = f"/World/Gantry/RobotCarriage_{i}"
            UsdGeom.Xform.Define(self.stage, c_path)
            self.carriage_paths.append(c_path)
# ================================
    # def _add_ur10(self):
    #     root = get_assets_root_path()
    #     # ur10_usd = root + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
    #     ur10_usd = "/home/rokey/isaacsim/ur10_mop2/ur10/ur10.usd"
    #     add_reference_to_stage(usd_path=ur10_usd, prim_path=self.robot_prim_path)
    #     UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(self.robot_prim_path)).SetTranslate(Gf.Vec3d(0, 0, self.rail_height + 0.3))
    #     self.robot = Robot(prim_path=self.robot_prim_path, name="ur10", articulation_controller=ArticulationController())
    #     self.world.scene.add(self.robot)
    #     self.ee_prim = XFormPrim(self.robot_prim_path + "/ee_link", name="ee_link")
    #     print("[UR10] robot added")
# ==================================================
    def _add_ur10(self):
        root = get_assets_root_path()
        # ur10_usd = root + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
        ur10_usd = "/home/rokey/isaacsim/ur10_mop2/ur10/ur10.usd"

        pitch = self.tank_size + self.spacing

        for i in range(self.num_robots):
            # 각 로봇 전용 캐리지 경로
            carriage_path = self.carriage_paths[i]

            # i번째 로봇이 담당할 수조 row를 단순히 i로 매핑 (grid_size == num_robots 가정)
            row_idx = min(i, self.grid_size - 1)
            _, row_y = self._tank_center(row_idx, 0)

            # 캐리지를 해당 row 중앙 위에 올려둠
            UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(carriage_path)) \
                .SetTranslate(Gf.Vec3d(0.0, row_y, self.rail_height + 0.2))

            # UR10 prim 경로: 각 캐리지 밑에 하나씩
            prim_path = f"{carriage_path}/UR10"
            self.robot_prim_paths.append(prim_path)

            # UR10 USD 참조 추가
            add_reference_to_stage(usd_path=ur10_usd, prim_path=prim_path)

            # Robot 객체 생성
            robot = Robot(
                prim_path=prim_path,
                name=f"ur10_{i}",
                articulation_controller=ArticulationController()
            )
            self.world.scene.add(robot)
            self.robots.append(robot)

            # EE prim (각 로봇별 ee_link)
            ee = XFormPrim(prim_path + "/ee_link", name=f"ee_link_{i}")
            self.ee_prims.append(ee)

        # 마스터(0번)를 기존 self.robot, self.ee_prim으로 둠 (기존 코드 호환용)
        if self.robots:
            self.robot = self.robots[self.master_robot_id]
            self.ee_prim = self.ee_prims[self.master_robot_id]
            print(f"[UR10] {self.num_robots} robots added (0번을 마스터로 사용)")

    def _lighting_cam(self):
        UsdLux.DomeLight.Define(self.stage, "/World/DomeLight").CreateIntensityAttr(1500)

    # ---------- gantry movement ----------
    # def _set_beam_x(self, x):
    #     prim = self.stage.GetPrimAtPath("/World/Gantry/YBeam")
    #     ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
    #     if not ops: UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(x, 0, self.rail_height + 0.1))
    #     else:
    #         pos = ops[0].Get(); ops[0].Set(Gf.Vec3d(x, pos[1], pos[2]))

    # def _set_carriage_xyz(self, x, y, z):
    #     prim = self.stage.GetPrimAtPath("/World/Gantry/RobotCarriage")
    #     ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
    #     if not ops: UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(x, y, z))
    #     else: ops[0].Set(Gf.Vec3d(x, y, z))
    #     # keep UR10 aligned under carriage
    #     UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(self.robot_prim_path)).SetTranslate(Gf.Vec3d(x, y, z+0.2))
    #     self.current_z = z


    # def move_gantry_linear(self, x, y, steps=80):
    #     sx, sy = self.current_x, self.current_y
    #     for i in range(steps):
    #         a = (i + 1) / steps
    #         cx, cy = sx + a * (x - sx), sy + a * (y - sy)
    #         self._set_beam_x(cx)
    #         self._set_carriage_xyz(cx, cy, self.current_z)
    #         self.world.step(render=True)
    #     self.current_x, self.current_y = x, y
    def _set_beam_x(self, x):
        prim = self.stage.GetPrimAtPath("/World/Gantry/YBeam")
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        if not ops: UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(x, 0, self.rail_height + 0.1))
        else:
            pos = ops[0].Get(); ops[0].Set(Gf.Vec3d(x, pos[1], pos[2]))

    def _set_carriage_xyz(self, x, y, z):
        """
        마스터 carriage는 (x, y, z)로 이동.
        나머지 carriage들은 row offset만큼 y를 이동해서 따라감.
        """
        for i, c_path in enumerate(self.carriage_paths):
            # 마스터 carriage는 그대로 이동
            if i == self.master_robot_id:
                final_y = y
            else:
                # 슬레이브 carriage는 자기 row 오프셋만큼 shift
                master_row_y = self.row_offsets[self.master_robot_id]
                my_row_y = self.row_offsets[i]
                offset = my_row_y - master_row_y

                final_y = y + offset    # ★ 핵심: row 차이에 따라 이동

            prim = self.stage.GetPrimAtPath(c_path)
            ops = UsdGeom.Xformable(prim).GetOrderedXformOps()

            pos = Gf.Vec3d(x, final_y, z)

            if not ops:
                UsdGeom.XformCommonAPI(prim).SetTranslate(pos)
            else:
                ops[0].Set(pos)

    # 마스터 기준 기록
        self.current_x = x
        self.current_y = y
        self.current_z = z

    def move_gantry_linear(self, x, y, steps=80):
        steps = int(steps)
        sx, sy = self.current_x, self.current_y

        for i in range(steps):
            a = (i+1)/steps
            cx = sx + a*(x - sx)
            cy = sy + a*(y - sy)

            # 빔 이동
            self._set_beam_x(cx)

            # multi carriage 이동
            self._set_carriage_xyz(cx, cy, self.current_z)

            self.world.step(render=True)

        self.current_x, self.current_y = x, y

    # ---------- IK (FD-Jacobian, position-only) ----------
    def _ee_pos(self) -> np.ndarray:
        p, _ = self.ee_prim.get_world_pose()
        return np.array(p, dtype=np.float64)

    # def _get_q(self) -> np.ndarray:
    #     return np.array(self.robot.get_joint_positions(), dtype=np.float64).reshape(-1)

    # def _set_q(self, q: np.ndarray):
    #     try: self.robot.set_joint_positions(np.array(q, dtype=np.float64).reshape(-1))
    #     except Exception: self.robot.apply_action({"joint_positions": np.array(q, dtype=np.float64).reshape(-1)})
    def _get_q(self) -> np.ndarray:
        """마스터 로봇(0번)의 joint 벡터를 반환."""
        if self.robots:
            robot = self.robots[self.master_robot_id]
        else:
            robot = self.robot
        return np.array(robot.get_joint_positions(), dtype=np.float64).reshape(-1)

    def _set_q(self, q: np.ndarray):
        """계산된 q를 모든 로봇에 동일하게 적용."""
        q = np.array(q, dtype=np.float64).reshape(-1)
        if self.robots:
            for robot in self.robots:
                try:
                    robot.set_joint_positions(q)
                except Exception:
                    robot.apply_action({"joint_positions": q})
        else:
            try:
                self.robot.set_joint_positions(q)
            except Exception:
                self.robot.apply_action({"joint_positions": q})

    def _joint_limits(self, dof: int) -> Tuple[np.ndarray, np.ndarray]:
        try:
            low, high = self.robot.get_dof_limits()
            low = np.array(low, dtype=np.float64).reshape(-1)
            high = np.array(high, dtype=np.float64).reshape(-1)
            if low.size == dof and high.size == dof: return low, high
        except Exception: pass
        return -np.ones(dof)*math.pi, np.ones(dof)*math.pi

    def _fd_jacobian(self, dof: int) -> Tuple[np.ndarray, int]:
        J = np.zeros((3, dof), dtype=np.float64)
        steps = 0
        q0 = self._get_q()
        for i in range(dof):
            dq = np.zeros_like(q0); dq[i] = FD_H
            self._set_q(q0 + dq);   [self.world.step(render=False) for _ in range(FD_STEPS_PER_COL)]; steps += FD_STEPS_PER_COL
            p_plus = self._ee_pos()
            self._set_q(q0 - dq);   [self.world.step(render=False) for _ in range(FD_STEPS_PER_COL)]; steps += FD_STEPS_PER_COL
            p_minus = self._ee_pos()
            self._set_q(q0);        [self.world.step(render=False) for _ in range(FD_STEPS_PER_COL)]; steps += FD_STEPS_PER_COL
            J[:, i] = (p_plus - p_minus) / (2.0 * FD_H)
        return J, steps

    @staticmethod
    def _dls(Jp: np.ndarray, e: np.ndarray) -> np.ndarray:
        JJt = Jp @ Jp.T
        return Jp.T @ np.linalg.solve(JJt + (LAMBDA*LAMBDA)*np.eye(3), KP*e)

    # ---------- visual target ----------
    def _ensure_target(self, pos, eps: float = 1e-4):
        pos = np.array(pos, dtype=np.float64)
        if self.target is None:
            self.target = VisualCuboid("/World/Target", "target", position=pos, size=TARGET_SIZE)
            self.world.scene.add(self.target)
        else:
            # ✅ 이전 위치와 거의 같으면 USD 업데이트 생략
            cur_pos, _ = self.target.get_world_pose()
            cur_pos = np.array(cur_pos, dtype=np.float64)
            if np.max(np.abs(cur_pos - pos)) > eps:
                self.target.set_world_pose(position=pos)
    # ---------- poses ----------
    def go_to_joint_pose(self, q_goal: np.ndarray, steps: int = POSE_STEPS):
        dof = self._get_q().size
        q_goal = np.array(q_goal, dtype=np.float64).reshape(-1)
        if q_goal.size != dof: return
        q0 = self._get_q()
        for i in range(steps):
            a = (i+1)/steps
            q_cmd = q0*(1.0 - a) + q_goal*a
            self._set_q(q_cmd)
            self.world.step(render=True)

    # ---------- target-follow primitive for one segment (x0 -> x1) ----------
    def _follow_segment(self, x0: float, x1: float, y_line: float, z_floor: float):
        """
        ▶ 더 빠른 버전: J 풀 FD는 드물게, 그 사이엔 Broyden rank-1로 저렴하게 갱신
        ▶ 타깃 USD 업데이트와 렌더 호출도 스로틀링
        """
        START_WAIT_SEC   = 0.4
        SEG_TIMEOUT_SEC  = 0.6
        BACKOFF_SPEED    = 0.06

        # 🔧 튜닝 포인트(더 공격적으로 올려도 됨)
        J_PERIOD_FULL    = 10    # FD 풀 재계산 주기(스텝)
        RENDER_PERIOD    = 10    # 렌더링 주기(스텝)
        TARGET_UPDATE_PD = 2     # 타깃 USD 업데이트 주기(스텝)

        dt   = self.world.get_physics_dt()
        dof  = self._get_q().size
        q_lo, q_hi = self._joint_limits(dof)

        target = np.array([x0, y_line, z_floor], dtype=np.float64)
        self._ensure_target(target)

        started   = False
        direction = np.sign(x1 - x0) if abs(x1 - x0) > 1e-9 else 0.0
        t_wait    = 0.0
        t_seg     = 0.0

        # ✅ 캐시에서 가져오기
        Jp  = self._J_cache["J"]
        q_p = self._J_cache["q"]
        p_p = self._J_cache["p"]

        iters = 0
        last_full_iter = -10**9

        while True:
            # 현재 상태
            p_cur = self._ee_pos()
            q_cur = self._get_q()
            dist  = np.linalg.norm((target + EE_OFFSET) - p_cur)

            # 게이트
            if not started and dist <= FOLLOW_START_DIST:
                started = True

            # 게이트 타임아웃 → 백오프
            if not started:
                t_wait += dt
                if t_wait >= START_WAIT_SEC:
                    step = min(BACKOFF_SPEED * dt, TARGET_MAX_STEP)
                    vec  = (p_cur - (target + EE_OFFSET))
                    n    = np.linalg.norm(vec)
                    if n > 1e-9:
                        target[0] += (vec[0]/n) * step
                        if direction >= 0:
                            target[0] = min(max(target[0], min(x0, x1)), max(x0, x1))
                        else:
                            target[0] = max(min(target[0], max(x0, x1)), min(x0, x1))

            # 타깃 진행 (USD 업데이트는 주기적으로만)
            moved_target = False
            if started and dist <= FOLLOW_HOLD_DIST and direction != 0.0:
                step   = min(TARGET_SPEED * dt, TARGET_MAX_STEP)
                remain = abs(x1 - target[0])
                dx     = direction * min(step, remain)
                if abs(dx) > 1e-6:
                    target[0] += dx
                    moved_target = True

            # 타깃 USD 업데이트 스로틀
            if moved_target or (iters % TARGET_UPDATE_PD == 0):
                self._ensure_target(target)

            # 종료 체크
            e = (target + EE_OFFSET) - p_cur
            if np.linalg.norm(e) < POS_TOL and (abs(x1 - target[0]) < 1e-4):
                break

            # ---- J 갱신 ----
            need_full = (Jp is None) or (iters - last_full_iter >= J_PERIOD_FULL)

            if need_full:
                # ⛏️ 풀 FD (render=False로만 돎)
                Jp, _ = self._fd_jacobian(dof)
                last_full_iter = iters
            else:
                # 🧠 Broyden rank-1 : J_{k+1} = J_k + ((y - J_k s) s^T) / (s^T s)
                # 여기서 s = (q_cur - q_p), y = (p_cur - p_p)
                if q_p is not None and p_p is not None:
                    s = (q_cur - q_p)
                    if np.any(np.abs(s) > 0):
                        y = (p_cur - p_p)
                        Js = Jp @ s
                        denom = float(s @ s)
                        if denom > 1e-12:
                            Jp = Jp + np.outer((y - Js), s) / denom

            # DLS 한 스텝
            dq = np.clip(self._dls(Jp, e), -DQ_LIMIT, DQ_LIMIT)
            q_cmd = np.clip(q_cur + dq * dt, q_lo, q_hi)
            self._set_q(q_cmd)

            # 캐시에 현재 상태 저장 (다음 Broyden용)
            self._J_cache["J"] = Jp
            self._J_cache["q"] = q_cur
            self._J_cache["p"] = p_cur

            # 렌더 스로틀
            self.world.step(render=(iters % RENDER_PERIOD == 0))
            iters += 1

            # 타임아웃
            t_seg += dt
            if t_seg >= SEG_TIMEOUT_SEC:
                print("[WARN] follow segment timeout; skipping remainder.")
                break

    def _follow_segment_2(self, x0: float, x1: float, y_line: float, z_floor: float):
        """
        안전 추종: 게이트 대기 타임아웃, 강제 시작, 세그먼트 타임아웃 포함.
        """
        # ---- 튜닝 파라미터 ----
        START_WAIT_SEC   = 0.4   # 게이트 대기 최대 시간
        SEG_TIMEOUT_SEC  = 0.6   # 세그먼트 최대 시간
        BACKOFF_SPEED    = 0.06  # 게이트 미충족 시 타깃을 EE쪽으로 천천히 이동
        # -----------------------

        dt   = self.world.get_physics_dt()
        dof  = self._get_q().size
        q_lo, q_hi = self._joint_limits(dof)

        target = np.array([x0, y_line, z_floor], dtype=np.float64)
        self._ensure_target(target)

        started = False
        direction = np.sign(x1 - x0) if abs(x1 - x0) > 1e-9 else 0.0
        t_wait = 0.0
        t_seg  = 0.0

        while True:
            p_cur = self._ee_pos()
            dist  = np.linalg.norm((target + EE_OFFSET) - p_cur)

            # 1) 게이트: 가까우면 정상 시작
            if not started and dist <= FOLLOW_START_DIST:
                started = True

            # 2) 게이트 타임아웃: 기다려도 가깝지 않으면 강제 시작(EE쪽으로 back-off)
            if not started:
                t_wait += dt
                if t_wait >= START_WAIT_SEC:
                    step = min(BACKOFF_SPEED * dt, TARGET_MAX_STEP)
                    # EE 쪽으로 한 스텝
                    vec  = (p_cur - (target + EE_OFFSET))
                    n    = np.linalg.norm(vec)
                    if n > 1e-9:
                        target[0] += (vec[0]/n) * step  # X만 이동(라인 유지)
                        # 클램프: 세그먼트 범위 밖으로 안 나가게
                        if direction >= 0:
                            target[0] = min(max(target[0], min(x0, x1)), max(x0, x1))
                        else:
                            target[0] = max(min(target[0], max(x0, x1)), min(x0, x1))
                        self._ensure_target(target)

            # 3) 타깃 진행(시작 상태이면서 너무 멀지 않을 때)
            if started and dist <= FOLLOW_HOLD_DIST and direction != 0.0:
                step = min(TARGET_SPEED * dt, TARGET_MAX_STEP)
                remain = abs(x1 - target[0])
                dx = direction * min(step, remain)
                target[0] += dx
                self._ensure_target(target)

            # 4) IK 한 스텝
            e = (target + EE_OFFSET) - p_cur
            if np.linalg.norm(e) < POS_TOL and (abs(x1 - target[0]) < 1e-4):
                break  # 정상 종료

            Jp, _ = self._fd_jacobian(dof)
            dq = np.clip(self._dls(Jp, e), -DQ_LIMIT, DQ_LIMIT)
            q = self._get_q()
            q_cmd = np.clip(q + dq * dt, q_lo, q_hi)
            self._set_q(q_cmd)
            self.world.step(render=True)

            # 5) 세그먼트 타임아웃 처리
            t_seg += dt
            if t_seg >= SEG_TIMEOUT_SEC:
                print("[WARN] follow segment timeout; skipping remainder.")
                break
    # def _follow_segment(self, x0: float, x1: float, y_line: float, z_floor: float):
    #     """Move target from x0 to x1 linearly with gating and have EE follow via IK."""
    #     dt = self.world.get_physics_dt()
    #     dof = self._get_q().size
    #     q_low, q_high = self._joint_limits(dof)

    #     # initialize at start
    #     target = np.array([x0, y_line, z_floor], dtype=np.float64)
    #     self._ensure_target(target)
    #     started = False
    #     done = False
    #     direction = np.sign(x1 - x0) if abs(x1 - x0) > 1e-9 else 0.0

    #     while not done:
    #         p_cur = self._ee_pos()
    #         dist = np.linalg.norm((target + EE_OFFSET) - p_cur)

    #         # gate: start moving only when close enough
    #         if not started and dist <= FOLLOW_START_DIST:
    #             started = True

    #         # move target if started and not too far away
    #         if started and dist <= FOLLOW_HOLD_DIST and direction != 0.0:
    #             step = min(TARGET_SPEED*dt, TARGET_MAX_STEP)
    #             remain = abs(x1 - target[0])
    #             dx = direction * min(step, remain)
    #             target[0] += dx
    #             self._ensure_target(target)

    #         # IK toward current moving/holding target
    #         e = (target + EE_OFFSET) - p_cur
    #         if np.linalg.norm(e) < POS_TOL and (not started or abs(x1 - target[0]) < 1e-4):
    #             done = True
    #         else:
    #             Jp, _ = self._fd_jacobian(dof)
    #             dq = np.clip(self._dls(Jp, e), -DQ_LIMIT, DQ_LIMIT)
    #             q = self._get_q()
    #             q_cmd = np.clip(q + dq*dt, q_low, q_high)
    #             self._set_q(q_cmd)

    #         self.world.step(render=True)

    # ---------- half scan (+X / -X) with follow ----------
    def _scan_half_from_wall(self, row, col, side: str):
        dt = self.world.get_physics_dt()
        cx, cy = self.tank_positions[(row, col)]
        inner_half = self.tank_size/2.0 - self.wall_thickness - WALL_MARGIN
        z_floor = self.wall_thickness + Z_FLOOR_CLEAR

        y_min = cy - inner_half + EDGE_CLEAR
        y_max = cy + inner_half - EDGE_CLEAR
        ys = np.arange(y_min+0.05, y_max + 1e-9, STEP_Y)

        if side == "+X":
            beam_x = cx + inner_half
            move_beam_x = beam_x +0.20
            x_min = max(cx - inner_half + EDGE_CLEAR, beam_x - ARM_REACH_X)
            x_max = cx + inner_half - EDGE_CLEAR + 0.1
            self.move_gantry_linear(move_beam_x, cy, steps=40)
            self.go_to_joint_pose(READY_POSE2, steps=POSE_STEPS)
        else:
            beam_x = cx - inner_half
            move_beam_x = beam_x -0.20
            x_min = cx - inner_half + EDGE_CLEAR -0.1
            x_max = min(cx + inner_half - EDGE_CLEAR, beam_x + ARM_REACH_X)
            self.move_gantry_linear(move_beam_x, cy, steps=40)
            self.go_to_joint_pose(READY_POSE, steps=POSE_STEPS)

        if x_max < x_min + 1e-6:
            print(f"[WARN] No X span for {side} (reach too short).")
            return

        xs = np.arange(x_min, x_max + 1e-9, STEP_X)

        # lock beam to wall side; start at tank center Y
        # self.move_gantry_linear(beam_x, cy, steps=40)
        self._set_beam_x(move_beam_x)

        for j, y_line in enumerate(ys):
            # move carriage along Y (arm does X)
            self.move_gantry_linear(move_beam_x, y_line, steps=30)

            # zig-zag order
            x_list = xs if (j % 2 == 0) else xs[::-1]
            
            # start point ensure target positioned
            self._ensure_target([x_list[0], y_line, z_floor])

            for k in range(0,len(x_list)-1,1):
                next_k = min(k+1, len(x_list)-1)
                self._follow_segment(x_list[k], x_list[next_k], y_line, z_floor)
                

            # small dwell per line
            for _ in range(max(0, int(LINE_DWELL / max(dt, 1e-6)))):
                self.world.step(render=True)

    # ---------- phased sequence ----------
    def clean_tank_phased(self, row, col):
        cx, cy = self.tank_positions[(row, col)]
        z_floor = self.wall_thickness + Z_FLOOR_CLEAR

        # 이동자세 → Transit 높이 → 수조로 이동 → Transit 해제 → 준비자세
        self.go_to_joint_pose(TRANSIT_POSE2, steps=POSE_STEPS)
        self.move_gantry_linear(cx, cy)
        self._ensure_target([cx, cy, z_floor])
        # self.go_to_joint_pose(READY_POSE2, steps=POSE_STEPS)

        # +X 반 청소
        self._scan_half_from_wall(row, col, side="+X")

        # 이동자세 → 준비자세 → −X 반 청소
        self.go_to_joint_pose(TRANSIT_POSE2, steps=POSE_STEPS)
        self.go_to_joint_pose(TRANSIT_POSE, steps=POSE_STEPS)
        # self.go_to_joint_pose(READY_POSE, steps=POSE_STEPS)
        self._scan_half_from_wall(row, col, side="-X")

        # 마지막 이동자세 → Transit 높이 ON (다음 탱크 전)
        self.go_to_joint_pose(TRANSIT_POSE, steps=POSE_STEPS)
        
        
    def hide_factoryenv_ceiling(self):
        """
        /World/FactoryEnv 아래 공장 천장 타일(SM_CeilingA_*)을 숨기는 함수.
        - RemovePrim() 대신 MakeInvisible()을 사용하여 레퍼런스 충돌 방지
        """

        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            name = prim.GetName()

            # FactoryEnv 안에 있는 Ceiling 프림만 대상으로
            if not path.startswith("/World/FactoryEnv/"):
                continue

            # 이름 패턴 매칭
            if "Ceiling" in name or "ceiling" in name or name.startswith("SM_CeilingA_"):
                try:
                    UsdGeom.Imageable(prim).MakeInvisible()
                    print(f"[HIDE] Ceiling hidden → {path}")
                except Exception as e:
                    print(f"[HIDE-ERR] {path} :: {e}")
                    
    def hide_specific_beams(self):
        """
        FactoryEnv 안의 특정 Beam(Mesh)만 숨김.
        사용자가 지정한 이름만 Invisible 처리.
        """
        from pxr import UsdGeom

        # 숨기려는 정확한 프림 경로들
        targets = [
            "/World/FactoryEnv/SM_BeamA_9M37/SM_BeamA_9M",
            "/World/FactoryEnv/SM_BeamA_9M38/SM_BeamA_9M",
        ]

        for path in targets:
            prim = self.stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                try:
                    UsdGeom.Imageable(prim).MakeInvisible()
                    print(f"[HIDE] Beam hidden → {path}")
                except Exception as e:
                    print(f"[ERR] Failed to hide {path}: {e}")
            else:
                print(f"[WARN] Prim not found: {path}")
        

    # ---------- build/run ----------
    def build(self):
        if self.stage.GetPrimAtPath("/World"):
            self.stage.RemovePrim("/World")
        root = get_assets_root_path()
        warehouse_usd = root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        add_reference_to_stage(warehouse_usd, "/World/FactoryEnv")
        self.hide_factoryenv_ceiling()
        self.hide_specific_beams()
        
        # self._ground()
        self._rails()
        self._add_ur10()
        self._lighting_cam()
#==============#========================
        self.create_all_tanks()
        self.setup_camera() 
        self.material_change()
        self.create_two_horizontal_beams()
        self.setup_row_cameras_on_gantry()

#==============#========================

    def run(self):
        self.build()
        self.world.reset()
#==============#========================
        self.item_contact_and_hide_water()
#==============#========================
        for _ in range(20): self.world.step(render=True)

        # column-major
        for c in range(self.grid_size):
            for r in range(0,self.grid_size,2):
                tx, ty = self.tank_positions[(r, c)]
                # 이미 Transit 높이 ON 상태로 들어옴
                self.move_gantry_linear(tx, ty)
                self.clean_tank_phased(r, c)  # 내부에서 마지막에 Transit ON으로 나감

        print("\nAll tanks cleaned (phased + follow). Idle…")
        while simulation_app.is_running():
            self.world.step(render=True)
        simulation_app.close()


if __name__ == "__main__":
    env = AquacultureGantrySystem()
    env.run()
