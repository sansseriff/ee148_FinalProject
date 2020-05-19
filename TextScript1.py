# this is used to control Blender to render 64 sub-frames for each training/testing example.


import bpy
import random
import math as m
import os


bpy.context.scene.render.resolution_x = 240
bpy.context.scene.render.resolution_x = 240

#vector pass does not work when motion blur is turned on!
bpy.context.scene.render.use_motion_blur = False
bpy.context.scene.cycles.max_bounces = 1
bpy.context.scene.cycles.diffuse_bounces = 1
bpy.context.scene.cycles.glossy_bounces = 1
bpy.context.scene.cycles.transparent_max_bounces = 1
bpy.context.scene.cycles.transmission_bounces = 1
bpy.context.scene.cycles.volume_bounces = 1
bpy.context.scene.cycles.use_animated_seed = True
bpy.context.scene.cycles.samples = 2

save = True

# Material.001 is a material made custom and pre-loaded into the .blend file.
mat = bpy.data.materials.get("Material.001")


for example in range(1000):
    print("############################################EX", example)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    bpy.context.scene.render.engine = 'CYCLES'
    #bpy.context.scene.cycles.feature_set = 'EXPERIMENTAL'
    bpy.context.scene.cycles.device = 'CPU'

    
    bpy.context.scene.cycles.samples = 2





    #set up the scene
    scene_brightness = .4
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (scene_brightness, scene_brightness, scene_brightness, 1)
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 10), rotation=(0, 0, 3.14159))
    context = bpy.context
    scene = context.scene
    currentCameraObj = bpy.data.objects[bpy.context.active_object.name]
    scene.camera = currentCameraObj
    bpy.context.object.data.lens = 116

    bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 0))
    sun1 = bpy.context.object
    sun1.rotation_euler = (2*m.pi*random.uniform(0,1), 2*m.pi*random.uniform(0,1), 2*m.pi*random.uniform(0,1))
    #sun1.energy = 5
    bpy.context.object.data.energy = 2
    
    
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 0))
    sun2 = bpy.context.object
    sun2.rotation_euler = (2*m.pi*random.uniform(0,1), 2*m.pi*random.uniform(0,1), 2*m.pi*random.uniform(0,1))
    #sun2.energy = 5
    bpy.context.object.data.energy = 2

    #pick random point on sphere radius 10
    rho = 5
    theta = 2*m.pi*random.uniform(0,1)
    phi = m.acos(2*random.uniform(0,1) - 1)

    #calcualte cartesian coordinates of it
    x_i = rho*m.sin(phi)*m.cos(theta)
    y_i = rho*m.sin(phi)*m.sin(theta)
    z_i = rho*m.cos(phi)


    location_initial = (x_i,y_i,z_i)
    location_final = (-x_i,-y_i,-z_i)


    delta_d = 2*(-x_i,-y_i,-z_i)
    bpy.ops.mesh.primitive_monkey_add(size=2, enter_editmode=False, location=location_initial)
    bpy.ops.object.shade_smooth()
    bpy.ops.object.material_slot_add()
    obj = bpy.context.object
    obj.name = "monkey" + str(int(random.uniform(0,10000)))
    
    
    
    obj.data.materials[0] = mat
    
    
   

    obj.rotation_euler = (2*m.pi*random.uniform(0,1), 2*m.pi*random.uniform(0,1), 2*m.pi*random.uniform(0,1))



    #smooth the mesh 
    obj.modifiers.new('My SubDiv', 'SUBSURF')
    obj.modifiers[0].levels = 1



    obj.keyframe_insert(data_path="location", frame=-3)
    obj.keyframe_insert(data_path="rotation_euler", frame=-3)

    obj.location = location_final
    obj.rotation_euler = (10*m.pi*random.uniform(0,1), 10*m.pi*random.uniform(0,1), 10*m.pi*random.uniform(0,1))


    obj.keyframe_insert(data_path="location", frame=3)
    obj.keyframe_insert(data_path="rotation_euler", frame=3)

    bpy.context.scene.frame_set(3)

    #slow down the scene by 1000x
    bpy.context.scene.render.frame_map_old = 1
    bpy.context.scene.render.frame_map_new = 1000

    bpy.data.scenes["Scene"].node_tree.nodes["File Output"].base_path = "C:\\data\\proof_data"
    renderFolder = "C:\data\proof_data"

    start_frame = 0
    end_frame = 0 + 64
    
    #bpy.data.scenes["Scene"].node_tree.nodes["File Output"].base_path ='C:\data\proof_data'
    #bpy.data.scenes["Scene"].node_tree.(null) = "Speed_" + str(example) + "_"
    #bpy.data.scenes["Scene"].node_tree.(null) = "Image_"+ str(example) + "_"

    nodes = bpy.data.scenes[0].node_tree.nodes 
    imgOutNode = nodes["File Output"] 
    
    imgOutNode.file_slots[0].path = "ex" + str(example) + "_Vector_"
    imgOutNode.file_slots[1].path = "ex" + str(example) + "_Img_"
    
    
    if save:
        for f in range(start_frame,end_frame + 1):
                scene.frame_set( f ) # Set frame

                #frmNum   = str( f ).zfill(5) # Formats 5 --> 005
                #fileName = "ex_{a}_frm_{f}".format(a = example, f = frmNum )
                fileName = "test"
                fileName += scene.render.file_extension
                bpy.context.scene.render.filepath = os.path.join( renderFolder, fileName )

                bpy.ops.render.render()
            






# bpy.ops.object.editmode_toggle()
# bpy.ops.uv.smart_project()
# bpy.ops.material.new()


