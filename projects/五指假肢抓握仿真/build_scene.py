"""Derive our experiment scene from immutable Apache-2.0 Shadow Hand assets."""
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
tree=ET.parse(ROOT/'upstream_shadow/right_hand.xml');r=tree.getroot()
r.set('model','PBL five-finger reflex grasp')
r.find('compiler').set('meshdir','upstream_shadow/assets')
opt=r.find('option');opt.set('timestep','0.001');opt.set('integrator','implicitfast')
opt.set('iterations','100');opt.set('noslip_iterations','5');opt.set('gravity','0 0 -9.81')
vis=ET.SubElement(r,'visual');ET.SubElement(vis,'global',offwidth='1000',offheight='700')
ET.SubElement(vis,'quality',shadowsize='2048');ET.SubElement(vis,'headlight',ambient='.4 .4 .4',diffuse='.7 .7 .7')
asset=r.find('asset')
ET.SubElement(asset,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.82 .88 .92',rgb2='.97 .98 .99',width='512',height='3072')
ET.SubElement(asset,'texture',name='floor',type='2d',builtin='checker',rgb1='.86 .9 .93',rgb2='.95 .96 .97',width='256',height='256')
ET.SubElement(asset,'material',name='floor_mat',texture='floor',texrepeat='5 5',reflectance='.08')
ET.SubElement(asset,'material',name='pad_teal',rgba='.12 .57 .63 1',specular='.3',shininess='.4')
ET.SubElement(asset,'material',name='object_orange',rgba='.96 .59 .16 1',specular='.25')
# Retain all 24 hand/wrist joints and all 20 original actuator transmissions.
world=r.find('worldbody');forearm=world.find('body');forearm.set('quat','1 0 0 0')
ET.SubElement(forearm,'joint',name='transport_lift',type='slide',axis='0 0 1',range='0 .14',damping='25',armature='.1')
for geom in forearm.iter('geom'):
    if geom.get('class')=='plastic_collision':
        geom.set('contype','1');geom.set('conaffinity','3');geom.set('friction','.6 .005 .0001')
        geom.set('solref','.02 1');geom.set('solimp','.95 .99 .002');geom.set('condim','6');geom.set('friction','.6 .015 .003')
    if geom.get('mesh') in ['f_distal_pst','th_distal_pst'] and geom.get('class')=='plastic_visual':geom.set('material','pad_teal')
act=r.find('actuator');ET.SubElement(act,'position',name='lift_motor',joint='transport_lift',kp='1500',kv='140',ctrlrange='0 .2',forcerange='-100 100')
for a in act:
    if a.get('name','').startswith('rh_A_WR'):a.set('kp','40');a.set('kv','1.5')
ET.SubElement(world,'light',pos='.4 -.6 1',dir='-.4 .6 -1',diffuse='.8 .8 .8')
ET.SubElement(world,'light',pos='-.3 .3 .8',dir='.3 -.3 -1',diffuse='.6 .6 .6')
ET.SubElement(world,'geom',name='floor',type='plane',size='1 1 .1',material='floor_mat',contype='4',conaffinity='2')
ET.SubElement(world,'geom',name='support',type='box',pos='.095 0 .332',size='.07 .16 .005',rgba='.43 .55 .62 1',contype='4',conaffinity='2',friction='.6 .005 .0001',solref='.02 1')
ET.SubElement(world,'geom',name='support_post',type='box',pos='.10 0 .166',size='.025 .045 .166',rgba='.43 .55 .62 1',contype='4',conaffinity='2',friction='.6 .005 .0001',solref='.02 1')
obj=ET.SubElement(world,'body',name='object',pos='.047 0 .3655')
ET.SubElement(obj,'freejoint',name='object_free')
ET.SubElement(obj,'geom',name='object_geom',type='capsule',size='.028 .047',quat='1 1 0 0',mass='.12',material='object_orange',contype='2',conaffinity='5',friction='.6 .015 .003',condim='6',solref='.02 1',solimp='.95 .99 .002')
ET.indent(tree,space='  ');tree.write(ROOT/'model.xml',encoding='utf-8',xml_declaration=True)
print('Derived scene saved: five digits, original hand joints and actuators retained.')
