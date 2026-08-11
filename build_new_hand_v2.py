"""Build robotis_5f_hand_v2 retargeting model from the new (rh_5_right_2) hand.
Minimal, robust: keep the old retargeting-URDF structure (world->palm->fingers + tips + palm_surface
+ MANO mapping), swap in the NEW hand's joint kinematics (origin/axis/limit) and the new mesh dir."""
import xml.etree.ElementTree as ET, shutil
from pathlib import Path

R = Path("/home/aalexiev/brl/Dexterous-Hand-Retargeting")
MS = Path("/home/aalexiev/brl/reflexive-rl/third_party/manipulator-software")
OLD = R/"assets/robots/robotis_5f_hand"
V2  = R/"assets/robots/robotis_5f_hand_v2"
NEW_URDF = MS/"assets/scene/flexiv_arm/urdf/Rizon4_new_hand.urdf"

HN=["a08_finger_r_joint_1_thumb1","a09_finger_r_joint_1_thumb2","a10_finger_r_joint_1_thumb3","a11_finger_r_joint_1_thumb4",
    "a12_finger_r_joint_2_index1","a13_finger_r_joint_2_index2","a14_finger_r_joint_2_index3","a15_finger_r_joint_2_index4",
    "a16_finger_r_joint_3_middle1","a17_finger_r_joint_3_middle2","a18_finger_r_joint_3_middle3","a19_finger_r_joint_3_middle4",
    "a20_finger_r_joint_4_ring1","a21_finger_r_joint_4_ring2","a22_finger_r_joint_4_ring3","a23_finger_r_joint_4_ring4",
    "a24_finger_r_joint_5_little1","a25_finger_r_joint_5_little2","a26_finger_r_joint_5_little3","a27_finger_r_joint_5_little4"]
suffix2full={h.split("_",1)[1]:h for h in HN}

# 1) fresh copy of the old model dir
if V2.exists(): shutil.rmtree(V2)
shutil.copytree(OLD, V2)
# 2) copy the new meshes (rh_5_right_2) into the v2 meshes dir
newmesh = MS/"assets/scene/flexiv_arm/robotis5f/meshes/rh_5_right_2"
shutil.copytree(newmesh, V2/"meshes/rh_5_right_2")

# 3) parse the NEW arm+hand URDF, index its hand joints by our a08.. names
new = ET.parse(NEW_URDF).getroot()
new_joints={}
for j in new.iter("joint"):
    nm=j.get("name")
    full=suffix2full.get(nm)
    if full: new_joints[full]=j

# 4) load the v2 URDF (old template) and overwrite joint kinematics + mesh dir
up = V2/"robotis_5f_hand.urdf"
tree=ET.parse(up); root=tree.getroot()
def set_or_replace(parent, tag, src):
    for e in parent.findall(tag): parent.remove(e)
    if src is not None:
        el=ET.SubElement(parent, tag)
        el.attrib.update(src.attrib)
n_upd=0
for j in root.iter("joint"):
    full=j.get("name")
    if full in new_joints:
        nj=new_joints[full]
        set_or_replace(j,"origin",nj.find("origin"))
        set_or_replace(j,"axis",  nj.find("axis"))
        set_or_replace(j,"limit", nj.find("limit"))
        n_upd+=1
tree.write(up)
# 5) mesh dir rh_5_right -> rh_5_right_2 (same basenames)
txt=up.read_text().replace("meshes/rh_5_right/","meshes/rh_5_right_2/")
up.write_text(txt)
# rename URDF + mapping to v2 names for clarity
(V2/"robotis_5f_hand.urdf").rename(V2/"robotis_5f_hand_v2.urdf")
print(f"built robotis_5f_hand_v2: updated {n_upd}/20 joints, meshes copied, mesh dir remapped")
print("files:", sorted(p.name for p in V2.iterdir()))
