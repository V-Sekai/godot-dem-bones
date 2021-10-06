/*************************************************************************/
/*  editor_scene_exporter_gltf_plugin.cpp                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#if TOOLS_ENABLED
#include "bake_blend_shapes_plugin.h"
#include "core/config/project_settings.h"
#include "core/error/error_list.h"
#include "core/object/object.h"
#include "core/templates/vector.h"
#include "dem_bones.h"
#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/check_box.h"
#include "scene/main/node.h"

String BakeBlendShapesPlugin::get_name() const {
	return "bake_blend_shapes";
}

bool BakeBlendShapesPlugin::has_main_screen() const {
	return false;
}

BakeBlendShapesPlugin::BakeBlendShapesPlugin(EditorNode *p_node) {
	editor = p_node;
	file_export_lib = memnew(EditorFileDialog);
	editor->get_gui_base()->add_child(file_export_lib);
	file_export_lib->set_title(TTR("Export Library"));
	file_export_lib->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	file_export_lib->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	file_export_lib->clear_filters();
	file_export_lib->add_filter("*.glb");
	file_export_lib->add_filter("*.gltf");
	file_export_lib->set_title(TTR("Bake Blend Shapes..."));
	String gltf_scene_name = TTR("Bake Blend Shapes");
	add_tool_menu_item(gltf_scene_name, callable_mp(this, &BakeBlendShapesPlugin::convert_scene_to_gltf2));
}

void BakeBlendShapesPlugin::convert_scene_to_gltf2() {
	Node *root = editor->get_tree()->get_edited_scene_root();
	if (!root) {
		editor->show_accept(TTR("This operation can't be done without a scene."), TTR("OK"));
		return;
	}

	List<String> deps;
	Ref<BlendShapeBake> bake;
	bake.instantiate();
	Error err = bake->convert_scene(root);
	if (err != OK) {
		ERR_PRINT(vformat("Bake scene error %s.", itos(err)));
	}
}

#endif // TOOLS_ENABLED
