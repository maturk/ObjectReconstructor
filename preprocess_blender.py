import bpy
import os
import glob
import numpy as np
from numpy.random import randint
import mathutils
from mathutils import Vector
from pathlib import Path

def random_poses(upper_lim = 10): # set of four camera views in each of eight quadrants around a model
    coords = []
    # quadrant 1
    x1 = randint(0,upper_lim,(1,4))
    y1 = randint(0,upper_lim,(1,4))
    z1 = randint(-upper_lim,upper_lim,(1,4))
    x2 = -1*randint(0,upper_lim,(1,4))
    y2 = randint(0,upper_lim,(1,4))
    z2 = randint(-upper_lim,upper_lim,(1,4))
    x3 = -1*randint(0,upper_lim,(1,4))
    y3 = -1*randint(0,upper_lim,(1,4))
    z3 = randint(-upper_lim,upper_lim,(1,4))
    x4 = randint(0,upper_lim,(1,4))
    y4 = -1*randint(0,upper_lim,(1,4))
    z4 = randint(-upper_lim,upper_lim,(1,4))
    for i in range(4):
        coords.append(mathutils.Vector((x1[0][i],y1[0][i],z1[0][i])))
        coords.append(mathutils.Vector((x2[0][i],y2[0][i],z2[0][i])))
        coords.append(mathutils.Vector((x3[0][i],y3[0][i],z3[0][i])))
        coords.append(mathutils.Vector((x4[0][i],y4[0][i],z4[0][i])))
    return coords

def update_camera(camera, focus_point=mathutils.Vector((0.0, 0.0, 0.0)), distance=2.5):
    looking_direction = camera.location - focus_point
    rot_quat = looking_direction.to_track_quat('Z', 'Y')

    camera.rotation_euler = rot_quat.to_euler()
    # Use * instead of @ for Blender <2.8
    camera.location = rot_quat @ mathutils.Vector((0.0, 0.0, distance))

# Setup
shapenet_directory = '/Users/maturk/data/Shapenet/'
save_directory = '/home/asl-student/mturkulainen/data/test'
print('start')
counter = 0

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = 'BLENDER_EEVEE'
render.image_settings.color_mode = 'RGBA'
render.image_settings.color_depth = '16' # ('8', '16') bits per channel
render.image_settings.file_format = 'PNG' # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = 640
render.resolution_y = 480
render.resolution_percentage = 100
render.film_transparent = True

scene.use_nodes = True

nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

# Clear default nodes
for n in nodes:
    nodes.remove(n)
    
# Create input render layer node
render_layers = nodes.new('CompositorNodeRLayers')

# Create depth output nodes
depth_file_output = nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
depth_file_output.base_path = ''
depth_file_output.file_slots[0].use_node_format = True
depth_file_output.format.file_format = 'PNG'
depth_file_output.format.color_depth = '16'
depth_file_output.format.color_mode = "BW"
# Remap as other types can not represent the full range of depth.
map = nodes.new(type="CompositorNodeMapValue")
# Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
map.offset = [0]
map.size = [0.25]
map.use_min = False
map.min = [0]


links.new(render_layers.outputs['Depth'], map.inputs[0])
links.new(map.outputs[0], depth_file_output.inputs[0])

# Add another light source so stuff facing away from light is not completely dark
#bpy.ops.object.light_add(type='SUN')
#light2 = bpy.data.lights['Sun']
#light2.use_shadow = False
#light2.specular_factor = 1.0
#light2.energy = 0.015
#bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Light'].rotation_euler
#bpy.data.objects['Sun'].rotation_euler[0] += 180


for folder in os.listdir(shapenet_directory):
    if not folder.startswith('.'):
        for object_dir in (os.listdir(os.path.join(shapenet_directory, folder))):
            model_path = os.path.join(shapenet_directory, folder, object_dir, 'models')
            models = glob.glob(os.path.join(model_path, '*.obj'))
            if models != []:
                model=models[0]
            else:
                model = []
            texture_paths = glob.glob(os.path.join(shapenet_directory, folder, object_dir, 'images', '*.jpg'))
            if texture_paths:
                for texture in texture_paths[0:4]:
                    texture_name = Path(texture).stem
                    bpy.ops.import_scene.obj(filepath=model,use_smooth_groups=False)
                    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
                    ob = bpy.context.active_object
                    mat = bpy.data.materials.new(name='custom material')
                    mat.use_nodes = True
                    bsdf = mat.node_tree.nodes["Principled BSDF"]
                    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
                    texImage.image = bpy.data.images.load(texture)
                    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])

                    if ob.data.materials:
                        ob.data.materials[0] = mat
                    else:
                        ob.data.materials.append(mat)
                    # Camera poses
                    poses = random_poses()
                    pose_num = 0
                    depths = []
                    for pose in poses:
                        bpy.data.objects['Camera'].location = pose
                        update_camera(bpy.data.objects['Camera'])
                        # Render color and depth
                        render_file_path = os.path.abspath(f"{save_directory}/{folder}/{object_dir}/{pose_num}_{texture_name}_color")
                        depth_file_path = os.path.abspath(f"{save_directory}/{folder}/{object_dir}/{pose_num}_{texture_name}_depth")
                        scene.render.filepath = render_file_path
                        depth_file_output.file_slots[0].path = depth_file_path + "_depth"
                        bpy.ops.render.render(write_still=True)  # render still
                        
                        pose_num+=1
                   
            elif model != []: # no texture available
                bpy.ops.import_scene.obj(filepath=model,use_smooth_groups=False)
                bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
                ob = bpy.context.active_object
                # Camera poses
                poses = random_poses()
                pose_num = 0
                depths = []
                for pose in poses:
                    bpy.data.objects['Camera'].location = pose
                    update_camera(bpy.data.objects['Camera'])
                    # Render color and depth
                    render_file_path = os.path.abspath(f"{save_directory}/{folder}/{object_dir}/{pose_num}_color")
                    scene.render.filepath = render_file_path
                    depth_file_path = os.path.abspath(f"{save_directory}/{folder}/{object_dir}/{pose_num}_depth")
                    depth_file_output.file_slots[0].path = depth_file_path + "_depth"
                    bpy.ops.render.render(write_still=True)  # render still

                    pose_num+=1
                
            counter +=1
            print(counter)
            
            # Delect objects and materials
            for o in bpy.context.scene.objects:
                if o.type == 'MESH':
                    o.select_set(True)
                    materials = list(o.data.materials)
                    print(materials)
                    if materials[0]:       
                        for i in range(len(materials)):
                            bpy.data.materials.remove(materials[i])
                else:
                    o.select_set(False)
            #Call the operator only once
            bpy.ops.object.delete()
            
