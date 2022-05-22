#include "dem_bones.h"

#include "scene/resources/importer_mesh.h"
#include "scene/3d/importer_mesh_instance_3d.h"

Error BlendShapeBake::convert_scene(Node *p_scene) {
	List<Node *> queue;
	AnimationPlayer *ap = nullptr;
	queue.push_back(p_scene);
	while (!queue.is_empty()) {
		List<Node *>::Element *front = queue.front();
		Node *node = front->get();
		ap = cast_to<AnimationPlayer>(node);
		if (ap) {
			queue.clear();
			break;
		}
		int child_count = node->get_child_count();
		for (int32_t i = 0; i < child_count; i++) {
			queue.push_back(node->get_child(i));
		}
		queue.pop_front();
	}
	if (!ap) {
		return OK;
	}
	queue.push_back(p_scene);

	List<StringName> animation_names;
	ap->get_animation_list(&animation_names);
	Vector<Ref<Animation>> animations;
	for (const StringName &name : animation_names) {
		Ref<Animation> anim = ap->get_animation(name);
		animations.push_back(anim);
	}
	while (!queue.is_empty()) {
		List<Node *>::Element *front = queue.front();
		Node *node = front->get();
		int child_count = node->get_child_count();
		for (int32_t i = 0; i < child_count; i++) {
			queue.push_back(node->get_child(i));
		}
		queue.pop_front();
		ImporterMeshInstance3D *mesh_instance_3d = cast_to<ImporterMeshInstance3D>(node);
		if (!mesh_instance_3d) {
			continue;
		}
		String skeleton_path = mesh_instance_3d->get_skeleton_path();
		Node *skeleton_node = mesh_instance_3d->get_node_or_null(skeleton_path);
		Skeleton3D *skeleton = cast_to<Skeleton3D>(skeleton_node);
		if (!skeleton) {
			continue;
		}
		// TODO 2021-04-20
		// - To hard-lock the transformations of bones: in the input mesh,
		// create bool attributes for joint nodes (bones) with name "demLock" and
		// set the value to "true".

		// TODO 2021-04-20
		// - To soft-lock skinning weights of vertices: in the input mesh,
		// paint per-vertex colors in gray-scale. The closer the color to white,
		// the more skinning weights of the vertex are preserved.
		Ref<ArrayMesh> surface_mesh = mesh_instance_3d->get_mesh();
		for (int32_t surface_i = 0; surface_i < surface_mesh->get_surface_count(); surface_i++) {
			Array surface_arrays = surface_mesh->surface_get_arrays(surface_i);
			Array blends_arrays = surface_mesh->surface_get_blend_shape_arrays(surface_i);
			int32_t new_blend_count = blends_arrays.size();
			int32_t blend_count = 0;
			if (surface_mesh.is_valid()) {
				for (int32_t surface_i = 0; surface_i < surface_mesh->get_surface_count();
						surface_i++) {
					blend_count += new_blend_count;
					NodePath mesh_track;
					// Dem::DemBonesExt<double, float> bones;
					// Vector<StringName> p_blend_paths;
					// String mesh_path = p_scene->get_path_to(mesh_instance_3d);
					// for (int32_t blend_i = 0; blend_i < surface_mesh->get_blend_shape_count(); blend_i++) {
					// 	String blend_name = surface_mesh->get_blend_shape_name(blend_i);
					// 	p_blend_paths.push_back(mesh_path + ":blend_shapes/" + blend_name);
					// }
					// Vector<StringName> p_bone_paths;
					// for (int32_t bone_i = 0; bone_i < skeleton->get_bone_count(); bone_i++) {
					// 	StringName bone_name = skeleton->get_bone_name(bone_i);
					// 	p_bone_paths.push_back(skeleton_path + ":" + bone_name);
					// }
					// Array bone_mesh = bones.convert(surface_arrays, blends_arrays, skeleton, p_blend_paths, p_bone_paths, animations);
				}
			}
		}
	}
	return OK;
}
