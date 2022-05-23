///////////////////////////////////////////////////////////////////////////////
//               Dem Bones - Skinning Decomposition Library                  //
//         Copyright (c) 2019, Electronic Arts. All rights reserved.         //
///////////////////////////////////////////////////////////////////////////////

#ifndef DEM_BONES_EXT
#define DEM_BONES_EXT

#include "DemBones.h"

#include "thirdparty/eigen/Eigen/Eigen/Geometry"
#include <stdint.h>

#ifndef DEM_BONES_MAT_BLOCKS
#include "MatBlocks.h"
#define DEM_BONES_DEM_BONES_EXT_MAT_BLOCKS_UNDEFINED
#endif

#include "core/config/engine.h"
#include "dem_bones.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/importer_mesh.h"
#include "scene/resources/mesh.h"

namespace Dem {
/**  @class DemBonesExt DemBonesExt.h "DemBones/DemBonesExt.h"
		@brief Extended class to handle hierarchical skeleton with local
   rotations/translations and bind matrices

		@details Call computeRTB() to get local rotations/translations and bind
   matrices after skinning decomposition is done and other data is set.

		@b _Scalar is the floating-point data type. @b _AniMeshScalar is the
   floating-point data type of mesh sequence #vertex.
*/
template <class _Scalar, class _AniMeshScalar>
class DemBonesExt : public DemBones<_Scalar, _AniMeshScalar> {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	using MatrixX = Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>;
	using Matrix4 = Eigen::Matrix<_Scalar, 4, 4>;
	using Matrix3 = Eigen::Matrix<_Scalar, 3, 3>;
	using VectorX = Eigen::Matrix<_Scalar, Eigen::Dynamic, 1>;
	using Vector4 = Eigen::Matrix<_Scalar, 4, 1>;
	using Vector3 = Eigen::Matrix<_Scalar, 3, 1>;
	using SparseMatrix = Eigen::SparseMatrix<_Scalar>;
	using Triplet = Eigen::Triplet<_Scalar>;

	using DemBones<_Scalar, _AniMeshScalar>::nIters;
	using DemBones<_Scalar, _AniMeshScalar>::nInitIters;
	using DemBones<_Scalar, _AniMeshScalar>::nTransIters;
	using DemBones<_Scalar, _AniMeshScalar>::transAffine;
	using DemBones<_Scalar, _AniMeshScalar>::transAffineNorm;
	using DemBones<_Scalar, _AniMeshScalar>::nWeightsIters;
	using DemBones<_Scalar, _AniMeshScalar>::nnz;
	using DemBones<_Scalar, _AniMeshScalar>::weightsSmooth;
	using DemBones<_Scalar, _AniMeshScalar>::weightsSmoothStep;
	using DemBones<_Scalar, _AniMeshScalar>::weightEps;

	using DemBones<_Scalar, _AniMeshScalar>::num_vertices;
	using DemBones<_Scalar, _AniMeshScalar>::num_bones;
	using DemBones<_Scalar, _AniMeshScalar>::num_subjects;
	using DemBones<_Scalar, _AniMeshScalar>::num_total_frames;
	using DemBones<_Scalar, _AniMeshScalar>::frame_start_index;
	using DemBones<_Scalar, _AniMeshScalar>::frame_subject_id;
	using DemBones<_Scalar, _AniMeshScalar>::rest_pose_geometry;
	using DemBones<_Scalar, _AniMeshScalar>::skinning_weights;
	using DemBones<_Scalar, _AniMeshScalar>::lock_weight;
	using DemBones<_Scalar, _AniMeshScalar>::bone_transform_mat;
	using DemBones<_Scalar, _AniMeshScalar>::lock_mat;
	using DemBones<_Scalar, _AniMeshScalar>::vertex;
	using DemBones<_Scalar, _AniMeshScalar>::fv;

	//! Timestamps for bone transformations #bone_transform_mat, [@c size] = #num_subjects, #fTime(@p k) is
	//! the timestamp of frame @p k
	Eigen::VectorXd fTime;

	//! Name of bones, [@c size] = #num_bones, #boneName(@p j) is the name bone of @p j
	std::vector<std::string> bone_name;

	//! Parent bone index, [@c size] = #num_bones, #parent(@p j) is the index of parent
	//! bone of @p j, #parent(@p j) = -1 if @p j has no parent.
	Eigen::VectorXi parent;

	//! Original bind pre-matrix, [@c size] = [4*#num_subjects, 4*#num_bones], #bind.@a block(4*@p
	//! s, 4*@p j, 4, 4) is the global bind matrix of bone @p j on subject @p s at
	//! the rest pose
	MatrixX bind;

	//! Inverse pre-multiplication matrices, [@c size] = [4*#num_subjects, 4*#num_bones],
	//! #preMulInv.@a block(4*@p s, 4*@p j, 4, 4) is the inverse of pre-local
	//! transformation of bone @p j on subject @p s
	MatrixX pre_mult_inv;

	//! Rotation order, [@c size] = [3*#num_subjects, #num_bones], #rotOrder.@a col(@p j).@a
	//! segment<3>(3*@p s) is the rotation order of bone @p j on subject @p s,
	//! 0=@c X, 1=@c Y, 2=@c Z, e.g. {0, 1, 2} is @c XYZ order
	Eigen::MatrixXi rot_order;

	//! Orientations of bones,  [@c size] = [3*#num_subjects, #num_bones], @p orient.@a col(@p
	//! j).@a segment<3>(3*@p s) is the(@c rx, @c ry, @c rz) orientation of bone
	//! @p j in degree
	MatrixX orient;

	//! Bind transformation update, 0=keep original, 1=set translations to p-norm
	//! centroids (using #transAffineNorm) and rotations to identity, 2=do 1 and
	//! group joints
	int bind_update = 0;

	int patience = 3;

	double tolerance = 1e-3;

	/** @brief Constructor and setting default parameters
	 */
	DemBonesExt();

	/** @brief Clear all data
	 */
	void clear();

	/** @brief Local rotations, translations and global bind matrices of a subject
		  @details Required all data in the base class: #rest_pose_geometry, #fv, #num_vertices, #vertex, #num_total_frames,
	 #frame_start_index, #frame_subject_id, #num_subjects, #bone_transform_mat, #skinning_weights, #num_bones

		  This function will initialize missing attributes:
		  - #parent: -1 vector (if no joint grouping) or parent to a root, [@c
	 size] = #num_bones
		  - #preMulInv: 4*4 identity matrix blocks, [@c size] = [4*#num_subjects, 4*#num_bones]
		  - #rotOrder: {0, 1, 2} vector blocks, [@c size] = [3*#num_subjects, #num_bones]
		  - #orient: 0 matrix, [@c size] = [3*#num_subjects, #num_bones]

		  @param[in] s is the subject index
		  @param[out] lr is the [3*@p nFr, #num_bones] by-reference output local
	 rotations, @p lr.@a col(@p j).segment<3>(3*@p k) is the (@c rx, @c ry, @c
	 rz) of bone @p j at frame @p k
		  @param[out] lt is the [3*@p nFr, #num_bones] by-reference output local
	 translations, @p lt.@a col(@p j).segment<3>(3*@p k) is the (@c tx, @c ty,
	 @c tz) of bone @p j at frame @p k
		  @param[out] gb is the [4, 4*#num_bones] by-reference output global bind
	 matrices, @p gb.@a block(0, 4*@p j, 4, 4) is the bind matrix of bone j
		  @param[out] lbr is the [3, #num_bones] by-reference output local rotations at
	 bind pose @p lbr.@a col(@p j).segment<3>(3*@p k) is the (@c rx, @c ry, @c
	 rz) of bone @p j
		  @param[out] lbt is the [3, #num_bones] by-reference output local translations
	 at bind pose, @p lbt.@a col(@p j).segment<3>(3*@p k) is the (@c tx, @c ty,
	 @c tz) of bone @p j
		  @param[in] degreeRot=true will output rotations in degree, otherwise
	 output in radian
  */
	void computeRTB(int s, MatrixX &r_local_rotations, MatrixX &r_local_translations, MatrixX &gb, MatrixX &local_bind_pose_rotation,
			MatrixX &r_local_bind_pose_translation, bool degreeRot = true);

private:
	void compute();

	/** p-norm centroids (using #transAffineNorm) and rotations to identity
		  @param s is the subject index
		  @param b is the [4, 4*#num_bones] by-reference output global bind matrices,
	 #b.#a block(0, 4*@p j, 4, 4) is the bind matrix of bone @p j
  */
	void compute_centroids(int s, MatrixX &b);

	/** Global bind pose
		  @param s is the subject index
		  @param bindUpdate is the type of bind pose update, 0=keep original, 1
	 or 2=set translations to p-norm centroids (using #transAffineNorm) and
	 rotations to identity
		  @param b is the the [4, 4*#num_bones] by-reference output global bind
	 matrices, #b.#a block(0, 4*@p j, 4, 4) is the bind matrix of bone @p j
  */
	void compute_bind(int p_subject, MatrixX &r_output_global_bind_matrix);

	/** Root joint
	 */
	int compute_root();

	/** Euler angles from rotation matrix
		  @param rMat is the 3*3 rotation matrix
		  @param curRot is the input current Euler angles, it is also the
	 by-reference output closet Euler angles correspond to @p rMat
		  @param ro is the rotation order, 0=@c X, 1=@c Y, 2=@c Z, e.g. {0, 1,
	 2} is @c XYZ order
		  @param eps is the epsilon
  */
	void to_rot(const Matrix3 &p_basis, Vector3 &r_input_euler, const Eigen::Vector3i &p_rotation_order,
			_Scalar p_epsilon = _Scalar(1e-10));
	struct Key {
		double time = 0.0; // time in secs
	};

	// transform key holds either Vector3 or Quaternion
	template <class T>
	struct TransformKey {
		::Vector3 loc;
		::Quaternion rot;
		::Vector3 scale;
	};

public:
	Array convert_blend_shapes_without_bones(Array p_mesh, ::PackedVector3Array p_vertex_array, HashMap<String, Vector<::Vector3>> p_blends, Vector<Ref<Animation>> p_anims);
};

template <class _Scalar, class _AniMeshScalar>
Dem::DemBonesExt<_Scalar, _AniMeshScalar>::DemBonesExt() :
		bind_update(0) {
	clear();
}

template <class _Scalar, class _AniMeshScalar>
void Dem::DemBonesExt<_Scalar, _AniMeshScalar>::clear() {
	fTime.resize(0);
	bone_name.resize(0);
	parent.resize(0);
	bind.resize(0, 0);
	pre_mult_inv.resize(0, 0);
	rot_order.resize(0, 0);
	orient.resize(0, 0);
	DemBones<_Scalar, _AniMeshScalar>::clear();
}

template <class _Scalar, class _AniMeshScalar>
void Dem::DemBonesExt<_Scalar, _AniMeshScalar>::computeRTB(int s, MatrixX &r_local_rotations, MatrixX &r_local_translations, MatrixX &gb, MatrixX &local_bind_pose_rotation, MatrixX &r_local_bind_pose_translation, bool degreeRot /*= true*/) {
	compute_bind(s, gb);

	if (parent.size() == 0) {
		if (bind_update == 2) {
			int root = compute_root();
			parent = Eigen::VectorXi::Constant(num_bones, root);
			parent(root) = -1;
		} else {
			parent = Eigen::VectorXi::Constant(num_bones, -1);
		}
	}
	if (pre_mult_inv.size() == 0) {
		pre_mult_inv = MatrixX::Identity(4, 4).replicate(num_subjects, num_bones);
	}
	if (rot_order.size() == 0) {
		rot_order = Eigen::Vector3i(0, 1, 2).replicate(num_subjects, num_bones);
	}
	if (orient.size() == 0) {
		orient = MatrixX::Zero(3 * num_subjects, num_bones);
	}

	int nFs = frame_start_index(s + 1) - frame_start_index(s);
	r_local_rotations.resize(nFs * 3, num_bones);
	r_local_translations.resize(nFs * 3, num_bones);
	local_bind_pose_rotation.resize(3, num_bones);
	r_local_bind_pose_translation.resize(3, num_bones);

	// #pragma omp parallel for
	for (int bone_i = 0; bone_i < num_bones; bone_i++) {
		Eigen::Vector3i ro = rot_order.col(bone_i).template segment<3>(s * 3);

		Vector3 ov = orient.vec3(s, bone_i) * EIGEN_PI / 180;
		Matrix3 invOM =
				Matrix3(Eigen::AngleAxis<_Scalar>(ov(ro(2)), Vector3::Unit(ro(2)))) *
				Eigen::AngleAxis<_Scalar>(ov(ro(1)), Vector3::Unit(ro(1))) *
				Eigen::AngleAxis<_Scalar>(ov(ro(0)), Vector3::Unit(ro(0)));
		invOM.transposeInPlace();

		Matrix4 lb;
		if (parent(bone_i) == -1) {
			lb = pre_mult_inv.blk4(s, bone_i) * gb.blk4(0, bone_i);
		} else {
			lb = pre_mult_inv.blk4(s, bone_i) * gb.blk4(0, parent(bone_i)).inverse() *
				 gb.blk4(0, bone_i);
		}

		Vector3 curRot = Vector3::Zero();
		to_rot(invOM * lb.template topLeftCorner<3, 3>(), curRot, ro);
		local_bind_pose_rotation.col(bone_i) = curRot;
		r_local_bind_pose_translation.col(bone_i) = lb.template topRightCorner<3, 1>();

		Matrix4 _lm;
		for (int k = 0; k < nFs; k++) {
			if (parent(bone_i) == -1) {
				_lm = pre_mult_inv.blk4(s, bone_i) * bone_transform_mat.blk4(k + frame_start_index(s), bone_i) * gb.blk4(0, bone_i);
			} else {
				_lm = pre_mult_inv.blk4(s, bone_i) *
					  (bone_transform_mat.blk4(k + frame_start_index(s), parent(bone_i)) * gb.blk4(0, parent(bone_i)))
							  .inverse() *
					  bone_transform_mat.blk4(k + frame_start_index(s), bone_i) * gb.blk4(0, bone_i);
			}
			to_rot(invOM * _lm.template topLeftCorner<3, 3>(), curRot, ro);
			r_local_rotations.vec3(k, bone_i) = curRot;
			r_local_translations.vec3(k, bone_i) = _lm.template topRightCorner<3, 1>();
		}
	}

	if (degreeRot) {
		r_local_rotations *= 180 / EIGEN_PI;
		local_bind_pose_rotation *= 180 / EIGEN_PI;
	}
}

template <class _Scalar, class _AniMeshScalar>
void Dem::DemBonesExt<_Scalar, _AniMeshScalar>::compute_centroids(int s, MatrixX &b) {
	MatrixX c = MatrixX::Zero(4, num_bones);
	for (int vert_i = 0; vert_i < num_vertices; vert_i++) {
		for (typename SparseMatrix::InnerIterator it(skinning_weights, vert_i); it; ++it) {
			c.col(it.row()) +=
					pow(it.value(), transAffineNorm) * rest_pose_geometry.vec3(s, vert_i).homogeneous();
		}
	}
	for (int bone_i = 0; bone_i < num_bones; bone_i++) {
		if ((c(3, bone_i) != 0) && (lock_mat(bone_i) == 0)) {
			b.transVec(0, bone_i) = c.col(bone_i).template head<3>() / c(3, bone_i);
		}
	}
}

template <class _Scalar, class _AniMeshScalar>
void Dem::DemBonesExt<_Scalar, _AniMeshScalar>::compute_bind(int p_subject, MatrixX &r_output_global_bind_matrix) {
	if (bind.size() == 0) {
		lock_mat = Eigen::VectorXi::Zero(num_bones);
		bind.resize(num_subjects * 4, num_bones * 4);
		for (int subject_i = 0; subject_i < num_subjects; subject_i++) {
			r_output_global_bind_matrix = MatrixX::Identity(4, 4).replicate(1, num_bones);
			compute_centroids(subject_i, r_output_global_bind_matrix);
			bind.block(4 * subject_i, 0, 4, 4 * num_bones) = r_output_global_bind_matrix;
		}
	}
	r_output_global_bind_matrix = bind.block(4 * p_subject, 0, 4, 4 * num_bones);
	if (bind_update >= 1) {
		compute_centroids(p_subject, r_output_global_bind_matrix);
	}
}

template <class _Scalar, class _AniMeshScalar>
int Dem::DemBonesExt<_Scalar, _AniMeshScalar>::compute_root() {
	VectorX err(num_bones);
	// #pragma omp parallel for
	for (int j = 0; j < num_bones; j++) {
		double ej = 0;
		for (int i = 0; i < num_vertices; i++) {
			for (int frame_i = 0; frame_i < num_total_frames; frame_i++) {
				ej += (bone_transform_mat.rotMat(frame_i, j) * rest_pose_geometry.vec3(frame_subject_id(frame_i), i) + bone_transform_mat.transVec(frame_i, j) -
						vertex.vec3(frame_i, i).template cast<_Scalar>())
							  .squaredNorm();
			}
		}
		err(j) = ej;
	}
	int rj;
	err.minCoeff(&rj);
	return rj;
}

template <class _Scalar, class _AniMeshScalar>
void Dem::DemBonesExt<_Scalar, _AniMeshScalar>::compute() {
	int np = patience;
	double prevErr = -1;
	DemBones<_Scalar, _AniMeshScalar>::init();
	for (int32_t iteration_i = 0; iteration_i < DemBonesExt<_Scalar, _AniMeshScalar>::nIters; iteration_i++) {
		DemBonesExt<_Scalar, _AniMeshScalar>::computeTranformations();
		DemBonesExt<_Scalar, _AniMeshScalar>::computeWeights();
		double err = DemBonesExt<_Scalar, _AniMeshScalar>::rmse();
		if ((err < prevErr * (1 + weightEps)) &&
				((prevErr - err) < DemBonesExt<_Scalar, _AniMeshScalar>::tolerance * prevErr)) {
			np--;
			if (np == 0) {
				print_line("Convergence has been reached.");
				break;
			}
		} else {
			np = patience;
		}
		prevErr = err;
		print_line("Root mean square error is " + rtos(prevErr));
	}
}

template <class _Scalar, class _AniMeshScalar>
void Dem::DemBonesExt<_Scalar, _AniMeshScalar>::to_rot(const Matrix3 &p_basis, Vector3 &r_input_euler, const Eigen::Vector3i &p_rotation_order, _Scalar p_epsilon /*= _Scalar(1e-10)*/) {
	Vector3 r0 = p_basis.eulerAngles(p_rotation_order(2), p_rotation_order(1), p_rotation_order(0)).reverse();
	_Scalar gMin = (r0 - r_input_euler).squaredNorm();
	Vector3 rMin = r0;
	Vector3 r;
	Matrix3 tmpMat;
	for (int fx = -1; fx <= 1; fx += 2) {
		for (_Scalar sx = -2 * EIGEN_PI; sx < 2.1 * EIGEN_PI; sx += EIGEN_PI) {
			r(0) = fx * r0(0) + sx;
			for (int fy = -1; fy <= 1; fy += 2) {
				for (_Scalar sy = -2 * EIGEN_PI; sy < 2.1 * EIGEN_PI;
						sy += EIGEN_PI) {
					r(1) = fy * r0(1) + sy;
					for (int fz = -1; fz <= 1; fz += 2) {
						for (_Scalar sz = -2 * EIGEN_PI; sz < 2.1 * EIGEN_PI;
								sz += EIGEN_PI) {
							r(2) = fz * r0(2) + sz;
							tmpMat =
									Matrix3(Eigen::AngleAxis<_Scalar>(r(p_rotation_order(2)),
											Vector3::Unit(p_rotation_order(2)))) *
									Eigen::AngleAxis<_Scalar>(r(p_rotation_order(1)), Vector3::Unit(p_rotation_order(1))) *
									Eigen::AngleAxis<_Scalar>(r(p_rotation_order(0)), Vector3::Unit(p_rotation_order(0)));
							if ((tmpMat - p_basis).squaredNorm() < p_epsilon) {
								_Scalar tmp = (r - r_input_euler).squaredNorm();
								if (tmp < gMin) {
									gMin = tmp;
									rMin = r;
								}
							}
						}
					}
				}
			}
		}
	}
	r_input_euler = rMin;
}

template <class _Scalar, class _AniMeshScalar>
Array Dem::DemBonesExt<_Scalar, _AniMeshScalar>::convert_blend_shapes_without_bones(Array p_mesh, ::PackedVector3Array p_vertex_array, HashMap<String, Vector<::Vector3>> p_blends, Vector<Ref<Animation>> p_anims) {
	if (!p_anims.size()) {
		return Array();
	}
	// TODO: 2021-10-05 Support multiple tracks by putting into one long track
	Ref<Animation> anim = p_anims[0];
	if (anim.is_null()) {
		return Array();
	}
	if (!p_blends.size()) {
		return p_mesh;
	}
	Vector<PackedVector3Array> blends;
	constexpr float FPS = 30.0f;
	int32_t new_frames = anim->get_length() * FPS;
	blends.resize(new_frames);
	for (int32_t frame_i = 0; frame_i < new_frames; frame_i++) {
		blends.write[frame_i] = p_vertex_array;
	}
	for (int32_t track_i = 0; track_i < anim->get_track_count(); track_i++) {
		String track_path = anim->track_get_path(track_i);
		Animation::TrackType track_type = anim->track_get_type(track_i);
		if (track_type != Animation::TYPE_BLEND_SHAPE) {
			continue;
		}
		const double increment = 1.0 / FPS;
		double time = 0.0;
		double length = anim->get_length();
		bool last = false;
		for (int32_t frame_i = 0; frame_i < new_frames; frame_i++) {
			float weight = 0.0f;
			if (anim->blend_shape_track_interpolate(track_i, time, &weight) != OK) {
				continue;
			}
			for (int32_t vertex_i = 0; vertex_i < p_blends[track_path].size(); vertex_i++) {
				blends.write[frame_i].write[vertex_i] = blends[frame_i][vertex_i].lerp(p_blends[track_path][vertex_i], weight);
			}
		}
	}
	num_total_frames = p_vertex_array.size();
	num_vertices = p_vertex_array.size();

	num_subjects = 1;
	num_total_frames = FPS * anim->get_length();
	fTime.resize(num_total_frames);
	frame_start_index.resize(num_subjects + 1);
	frame_start_index(0) = 0;
	for (int s = 0; s < num_subjects; s++) {
		for (int32_t subjects_i = 0; subjects_i < num_subjects; subjects_i++) {
			frame_start_index[s + 1] = frame_start_index[s] + num_total_frames; // TODO: fire 2022-05-22 Support different start frame
		}
	}

	frame_subject_id.resize(num_total_frames);
	for (int s = 0; s < num_subjects; s++) {
		for (int k = frame_start_index(s); k < frame_start_index(s + 1); k++) {
			frame_subject_id[k] = s;
		}
	}

	vertex.resize(3 * num_total_frames, num_vertices);
	for (int32_t frame_i = 0; frame_i < num_total_frames; frame_i++) {
		for (int32_t vertex_i = 0; vertex_i < blends[frame_i].size(); vertex_i++) {
			const float &x = blends[frame_i][vertex_i].x;
			const float &y = blends[frame_i][vertex_i].y;
			const float &z = blends[frame_i][vertex_i].z;
			fTime[frame_i] = double(frame_i * 1.0 / FPS);
			vertex.col(vertex_i).segment(frame_i * 3, 3) << x, y, z;
		}
	}

	PackedInt32Array indices = p_mesh[Mesh::ARRAY_INDEX];

	// Assume the mesh is a triangle mesh.
	const int indices_in_tri = 3;
	fv.resize(indices.size() / 3);
	for (int32_t index_i = 0; index_i < indices.size(); index_i += 3) {
		std::vector<int> polygon_indices;
		polygon_indices.resize(indices_in_tri);
		polygon_indices[0] = indices[index_i / 3 + 0];
		polygon_indices[1] = indices[index_i / 3 + 1];
		polygon_indices[2] = indices[index_i / 3 + 2];
		fv[index_i / 3] = polygon_indices;
	}
	// TODO: fire 2022-05-22 subjects > 1;
	rest_pose_geometry.resize(3, num_vertices);
	for (int32_t vertex_i = 0; vertex_i < p_vertex_array.size();
			vertex_i++) {
		float pos_x = p_vertex_array[vertex_i].x;
		float pos_y = p_vertex_array[vertex_i].y;
		float pos_z = p_vertex_array[vertex_i].z;
		rest_pose_geometry.col(vertex_i) << pos_x, pos_y, pos_z;
	}
	nnz = 4;
	bind_update = 2;
	nIters = 100;
	num_bones = 20;
	nInitIters = 20;
	DemBonesExt<_Scalar, _AniMeshScalar>::init();
	DemBonesExt<_Scalar, _AniMeshScalar>::compute();
	bool needCreateJoints = (bone_name.size() == 0);
	double radius;
	if (needCreateJoints) {
		bone_name.resize(num_bones);
		for (int j = 0; j < num_bones; j++) {
			std::ostringstream s;
			s << "joint" << j;
			bone_name[j] = s.str();
		}
		radius = sqrt((rest_pose_geometry - (rest_pose_geometry.rowwise().sum() / num_vertices).replicate(1, num_vertices)).cwiseAbs().rowwise().maxCoeff().squaredNorm() / num_subjects);
	}

	print_line(vformat("The number of bones %d", bone_name.size()));
	for (int s = 0; s < num_subjects; s++) {
		MatrixX local_rotations;
		MatrixX local_translations;
		MatrixX gb;
		MatrixX local_bind_pose_rotation;
		MatrixX local_bind_pose_translation;
		bool degree_rot = true;
		DemBonesExt<_Scalar, _AniMeshScalar>::computeRTB(s, local_rotations, local_translations, gb, local_bind_pose_rotation, local_bind_pose_translation, degree_rot);
		// if (needCreateJoints) {
		// 	exporter.createJoints(model.boneName, model.parent, radius);
		// }
		// exporter.setJoints(model.boneName, model.fTime.segment(model.fStart(s), model.fStart(s + 1) - model.fStart(s)), lr, lt, lbr, lbt);
		// exporter.setSkinCluster(model.boneName, model.w, gb);
	}
	// model.w=(wd/model.nS).sparseView(1, 1e-20);
	// model.lockW/=(double)model.nS;
	// if (!hasKeyFrame) model.m.resize(0, 0);

	print_line(vformat("Number of vertices %d", num_vertices));
	print_line(vformat("Number of joints %d", num_bones));
	print_line(vformat("Number of frames %d", num_total_frames));
	print_line(vformat("Number of skinning weights %d", skinning_weights.size()));
	return p_mesh;
}
} // namespace Dem
#ifdef DEM_BONES_DEM_BONES_EXT_MAT_BLOCKS_UNDEFINED
#undef blk4
#undef rotMat
#undef transVec
#undef vec3
#undef DEM_BONES_MAT_BLOCKS
#endif

#undef rotMatFromEuler

#endif
