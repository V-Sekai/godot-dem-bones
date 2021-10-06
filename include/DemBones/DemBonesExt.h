///////////////////////////////////////////////////////////////////////////////
//               Dem Bones - Skinning Decomposition Library                  //
//         Copyright (c) 2019, Electronic Arts. All rights reserved.         //
///////////////////////////////////////////////////////////////////////////////

#ifndef DEM_BONES_EXT
#define DEM_BONES_EXT

#include "DemBones.h"

#include <stdint.h>
#include <Eigen/Geometry>

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
	std::vector<std::string> boneName;

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
	MatrixX preMulInv;

	//! Rotation order, [@c size] = [3*#num_subjects, #num_bones], #rotOrder.@a col(@p j).@a
	//! segment<3>(3*@p s) is the rotation order of bone @p j on subject @p s,
	//! 0=@c X, 1=@c Y, 2=@c Z, e.g. {0, 1, 2} is @c XYZ order
	Eigen::MatrixXi rotOrder;

	//! Orientations of bones,  [@c size] = [3*#num_subjects, #num_bones], @p orient.@a col(@p
	//! j).@a segment<3>(3*@p s) is the(@c rx, @c ry, @c rz) orientation of bone
	//! @p j in degree
	MatrixX orient;

	//! Bind transformation update, 0=keep original, 1=set translations to p-norm
	//! centroids (using #transAffineNorm) and rotations to identity, 2=do 1 and
	//! group joints
	int bindUpdate;

	/** @brief Constructor and setting default parameters
   */
	DemBonesExt() :
			bindUpdate(0) { clear(); }

	/** @brief Clear all data
   */
	void clear() {
		fTime.resize(0);
		boneName.resize(0);
		parent.resize(0);
		bind.resize(0, 0);
		preMulInv.resize(0, 0);
		rotOrder.resize(0, 0);
		orient.resize(0, 0);
		DemBones<_Scalar, _AniMeshScalar>::clear();
	}

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
	void computeRTB(int s, MatrixX &lr, MatrixX &lt, MatrixX &gb, MatrixX &lbr,
			MatrixX &lbt, bool degreeRot = true) {
		computeBind(s, gb);

		if (parent.size() == 0) {
			if (bindUpdate == 2) {
				int root = computeRoot();
				parent = Eigen::VectorXi::Constant(num_bones, root);
				parent(root) = -1;
			} else
				parent = Eigen::VectorXi::Constant(num_bones, -1);
		}
		if (preMulInv.size() == 0)
			preMulInv = MatrixX::Identity(4, 4).replicate(num_subjects, num_bones);
		if (rotOrder.size() == 0)
			rotOrder = Eigen::Vector3i(0, 1, 2).replicate(num_subjects, num_bones);
		if (orient.size() == 0)
			orient = MatrixX::Zero(3 * num_subjects, num_bones);

		int nFs = frame_start_index(s + 1) - frame_start_index(s);
		lr.resize(nFs * 3, num_bones);
		lt.resize(nFs * 3, num_bones);
		lbr.resize(3, num_bones);
		lbt.resize(3, num_bones);

		// #pragma omp parallel for
		for (int j = 0; j < num_bones; j++) {
			Eigen::Vector3i ro = rotOrder.col(j).template segment<3>(s * 3);

			Vector3 ov = orient.vec3(s, j) * EIGEN_PI / 180;
			Matrix3 invOM =
					Matrix3(Eigen::AngleAxis<_Scalar>(ov(ro(2)), Vector3::Unit(ro(2)))) *
					Eigen::AngleAxis<_Scalar>(ov(ro(1)), Vector3::Unit(ro(1))) *
					Eigen::AngleAxis<_Scalar>(ov(ro(0)), Vector3::Unit(ro(0)));
			invOM.transposeInPlace();

			Matrix4 lb;
			if (parent(j) == -1)
				lb = preMulInv.blk4(s, j) * gb.blk4(0, j);
			else
				lb = preMulInv.blk4(s, j) * gb.blk4(0, parent(j)).inverse() *
					 gb.blk4(0, j);

			Vector3 curRot = Vector3::Zero();
			toRot(invOM * lb.template topLeftCorner<3, 3>(), curRot, ro);
			lbr.col(j) = curRot;
			lbt.col(j) = lb.template topRightCorner<3, 1>();

			Matrix4 _lm;
			for (int k = 0; k < nFs; k++) {
				if (parent(j) == -1)
					_lm = preMulInv.blk4(s, j) * bone_transform_mat.blk4(k + frame_start_index(s), j) * gb.blk4(0, j);
				else
					_lm = preMulInv.blk4(s, j) *
						  (bone_transform_mat.blk4(k + frame_start_index(s), parent(j)) * gb.blk4(0, parent(j)))
								  .inverse() *
						  bone_transform_mat.blk4(k + frame_start_index(s), j) * gb.blk4(0, j);
				toRot(invOM * _lm.template topLeftCorner<3, 3>(), curRot, ro);
				lr.vec3(k, j) = curRot;
				lt.vec3(k, j) = _lm.template topRightCorner<3, 1>();
			}
		}

		if (degreeRot) {
			lr *= 180 / EIGEN_PI;
			lbr *= 180 / EIGEN_PI;
		}
	}

private:
	/** p-norm centroids (using #transAffineNorm) and rotations to identity
          @param s is the subject index
          @param b is the [4, 4*#num_bones] by-reference output global bind matrices,
     #b.#a block(0, 4*@p j, 4, 4) is the bind matrix of bone @p j
  */
	void computeCentroids(int s, MatrixX &b) {
		MatrixX c = MatrixX::Zero(4, num_bones);
		for (int i = 0; i < num_vertices; i++)
			for (typename SparseMatrix::InnerIterator it(skinning_weights, i); it; ++it)
				c.col(it.row()) +=
						pow(it.value(), transAffineNorm) * rest_pose_geometry.vec3(s, i).homogeneous();
		for (int j = 0; j < num_bones; j++)
			if ((c(3, j) != 0) && (lock_mat(j) == 0))
				b.transVec(0, j) = c.col(j).template head<3>() / c(3, j);
	}

	/** Global bind pose
          @param s is the subject index
          @param bindUpdate is the type of bind pose update, 0=keep original, 1
     or 2=set translations to p-norm centroids (using #transAffineNorm) and
     rotations to identity
          @param b is the the [4, 4*#num_bones] by-reference output global bind
     matrices, #b.#a block(0, 4*@p j, 4, 4) is the bind matrix of bone @p j
  */
	void computeBind(int s, MatrixX &b) {
		if (bind.size() == 0) {
			lock_mat = Eigen::VectorXi::Zero(num_bones);
			bind.resize(num_subjects * 4, num_bones * 4);
			for (int k = 0; k < num_subjects; k++) {
				b = MatrixX::Identity(4, 4).replicate(1, num_bones);
				computeCentroids(k, b);
				bind.block(4 * k, 0, 4, 4 * num_bones) = b;
			}
		}
		b = bind.block(4 * s, 0, 4, 4 * num_bones);
		if (bindUpdate >= 1) {
			computeCentroids(s, b);
		}
	}

	/** Root joint
   */
	int computeRoot() {
		VectorX err(num_bones);
		// #pragma omp parallel for
		for (int j = 0; j < num_bones; j++) {
			double ej = 0;
			for (int i = 0; i < num_vertices; i++)
				for (int k = 0; k < num_total_frames; k++)
					ej += (bone_transform_mat.rotMat(k, j) * rest_pose_geometry.vec3(frame_subject_id(k), i) + bone_transform_mat.transVec(k, j) -
							vertex.vec3(k, i).template cast<_Scalar>())
								  .squaredNorm();
			err(j) = ej;
		}
		int rj;
		err.minCoeff(&rj);
		return rj;
	}

	/** Euler angles from rotation matrix
          @param rMat is the 3*3 rotation matrix
          @param curRot is the input current Euler angles, it is also the
     by-reference output closet Euler angles correspond to @p rMat
          @param ro is the rotation order, 0=@c X, 1=@c Y, 2=@c Z, e.g. {0, 1,
     2} is @c XYZ order
          @param eps is the epsilon
  */
	void toRot(const Matrix3 &rMat, Vector3 &curRot, const Eigen::Vector3i &ro,
			_Scalar eps = _Scalar(1e-10)) {
		Vector3 r0 = rMat.eulerAngles(ro(2), ro(1), ro(0)).reverse();
		_Scalar gMin = (r0 - curRot).squaredNorm();
		Vector3 rMin = r0;
		Vector3 r;
		Matrix3 tmpMat;
		for (int fx = -1; fx <= 1; fx += 2)
			for (_Scalar sx = -2 * EIGEN_PI; sx < 2.1 * EIGEN_PI; sx += EIGEN_PI) {
				r(0) = fx * r0(0) + sx;
				for (int fy = -1; fy <= 1; fy += 2)
					for (_Scalar sy = -2 * EIGEN_PI; sy < 2.1 * EIGEN_PI;
							sy += EIGEN_PI) {
						r(1) = fy * r0(1) + sy;
						for (int fz = -1; fz <= 1; fz += 2)
							for (_Scalar sz = -2 * EIGEN_PI; sz < 2.1 * EIGEN_PI;
									sz += EIGEN_PI) {
								r(2) = fz * r0(2) + sz;
								tmpMat =
										Matrix3(Eigen::AngleAxis<_Scalar>(r(ro(2)),
												Vector3::Unit(ro(2)))) *
										Eigen::AngleAxis<_Scalar>(r(ro(1)), Vector3::Unit(ro(1))) *
										Eigen::AngleAxis<_Scalar>(r(ro(0)), Vector3::Unit(ro(0)));
								if ((tmpMat - rMat).squaredNorm() < eps) {
									_Scalar tmp = (r - curRot).squaredNorm();
									if (tmp < gMin) {
										gMin = tmp;
										rMin = r;
									}
								}
							}
					}
			}
		curRot = rMin;
	}
	struct Key {
		double time = 0.0; // time in secs
	};

	// transform key holds either Vector3 or Quaternion
	template <class T>
	struct TKey : public Key {
		T value;
	};
	struct TransformKey {
		::Vector3 loc;
		::Quaternion rot;
		::Vector3 scale;
	};
	struct BlendKey {
		real_t weight;
	};

public:
	Array convert(Array p_mesh, Array p_blends, Skeleton3D *p_skeleton, Vector<StringName> p_blend_paths, Vector<StringName> p_bone_paths,
			Vector<Ref<Animation>> p_anims) {
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

		Map<StringName, Vector<TKey<TransformKey>>> transforms;
		Map<StringName, Vector<TKey<BlendKey>>> blends;
		float FPS = 30.0f;
		// TODO: Optimize
		for (int32_t track_i = 0; track_i < anim->get_track_count(); track_i++) {
			String track_path = anim->track_get_path(track_i);
			Animation::TrackType track_type = anim->track_get_type(track_i);
			if (track_type == Animation::TYPE_TRANSFORM3D) {
				const double increment = 1.0 / FPS;
				double time = 0.0;
				double length = anim->get_length();

				::Vector3 base_loc;
				::Quaternion base_rot;
				::Vector3 base_scale = ::Vector3(1, 1, 1);

				anim->transform_track_interpolate(track_i, 0.0f, &base_loc, &base_rot, &base_scale);

				bool last = false;
				Vector<Dem::DemBonesExt<double, float>::TKey<Dem::DemBonesExt<double, float>::TransformKey>> transform_anims;
				while (true) {
					::Vector3 loc = base_loc;
					::Quaternion rot = base_rot;
					::Vector3 scale = base_scale;

					anim->transform_track_interpolate(track_i, time, &loc, &rot, &scale);
					Dem::DemBonesExt<double, float>::TKey<Dem::DemBonesExt<double, float>::TransformKey> key;
					key.time = time;
					TransformKey transform_key;
					transform_key.loc = loc;
					transform_key.rot = rot;
					transform_key.scale = scale;
					key.value = transform_key;
					transform_anims.push_back(key);
					if (last) {
						break;
					}
					time += increment;
					if (time >= length) {
						last = true;
						time = length;
					}
					transforms.insert(track_path, transform_anims);
				}
			} else if (track_type == Animation::TYPE_VALUE) {
				const double increment = 1.0 / FPS;
				double time = 0.0;
				double length = anim->get_length();

				float base_weight = 0.0f;
				base_weight = anim->value_track_interpolate(track_i, 0.0f);

				bool last = false;
				Vector<TKey<BlendKey>> blend_anims;
				while (true) {
					float weight = base_weight;

					weight = anim->value_track_interpolate(track_i, time);
					TKey<BlendKey> key;
					key.time = time;
					BlendKey blend_key;
					blend_key.weight = weight;
					key.value = blend_key;
					blend_anims.push_back(key);
					if (last) {
						break;
					}
					time += increment;
					if (time >= length) {
						last = true;
						time = length;
					}
				}
				blends.insert(track_path, blend_anims);
			}
		}
		ERR_FAIL_NULL_V(p_skeleton, Array());
		num_subjects = 1;
		fTime.resize(FPS * anim->get_length());
		num_total_frames = fTime.size();
		frame_start_index.resize(num_subjects + 1);
		frame_start_index(0) = 0;

		for (int s = 0; s < num_subjects; s++) {
			frame_start_index(s + 1) = frame_start_index(s) + fTime.size();
		}

		frame_subject_id.resize(num_total_frames);
		for (int s = 0; s < num_subjects; s++) {
			for (int k = frame_start_index(s); k < frame_start_index(s + 1); k++) {
				frame_subject_id(k) = s;
			}
		}

		rest_pose_geometry.resize(num_subjects * 3, num_vertices);
		{
			PackedVector3Array vertex_arrays = p_mesh[Mesh::ARRAY_VERTEX];
			num_vertices = vertex_arrays.size();
		    rest_pose_geometry.resize(num_subjects * 3, num_vertices);
			for (int32_t vertex_i = 0; vertex_i < vertex_arrays.size();
					vertex_i++) {
				float pos_x = vertex_arrays[vertex_i].x;
				float pos_y = vertex_arrays[vertex_i].y;
				float pos_z = vertex_arrays[vertex_i].z;
				rest_pose_geometry.col(vertex_i) << pos_x, pos_y, pos_z;
			}
		}
		vertex.resize(3, num_vertices * num_total_frames);
		for (int32_t frame_i = 0; frame_i < num_total_frames; frame_i++) {
			PackedVector3Array blend_vertex_arrays = p_mesh[Mesh::ARRAY_VERTEX];
			for (int32_t blend_path_i = 0; blend_path_i < p_blend_paths.size(); blend_path_i++) {
				String blend_path = p_blend_paths[blend_path_i];
				if (!blends.has(blend_path)) {
					continue;
				}
				Vector<TKey<BlendKey>> keys = blends[blend_path];
				const Array &current_blend_array = p_blends[blend_path_i];
				const PackedVector3Array &blend = current_blend_array[Mesh::ARRAY_VERTEX];
				for (const TKey<BlendKey> &key : keys) {
					// #pragma omp parallel for
					for (int32_t vertex_i = 0; vertex_i < blend_vertex_arrays.size();
							vertex_i++) {
						BlendKey blend_key = key.value;
						float &pos_x = blend_vertex_arrays.write[vertex_i].x;
						const float &blend_pos_x = blend[vertex_i].x;
						float &pos_y = blend_vertex_arrays.write[vertex_i].y;
						const float &blend_pos_y = blend[vertex_i].y;
						float &pos_z = blend_vertex_arrays.write[vertex_i].z;
						const float &blend_pos_z = blend[vertex_i].z;
						pos_x = Math::lerp(pos_x, blend_pos_x, blend_key.weight);
						pos_y = Math::lerp(pos_y, blend_pos_y, blend_key.weight);
						pos_z = Math::lerp(pos_z, blend_pos_z, blend_key.weight);
					}
				}
			}
			// #pragma omp parallel for
			for (int32_t vertex_i = 0; vertex_i < blend_vertex_arrays.size();
					vertex_i++) {
				const float &pos_x = blend_vertex_arrays.write[vertex_i].x;
				const float &pos_y = blend_vertex_arrays.write[vertex_i].y;
				const float &pos_z = blend_vertex_arrays.write[vertex_i].z;
				vertex.col((vertex_i * frame_i) + vertex_i) << pos_x, pos_y, pos_z;
			}
		}
		PackedInt32Array indices = p_mesh[Mesh::ARRAY_INDEX];

		// Assume triangles
		const int indices_in_tri = 3;
		fv.resize(indices.size() / indices_in_tri);
		for (int32_t index_i = 0; index_i < indices.size(); index_i += 3) {
			std::vector<int> polygon_indices;
			polygon_indices.resize(indices_in_tri);
			polygon_indices[0] = indices[index_i / 3 + 0];
			polygon_indices[1] = indices[index_i / 3 + 1];
			polygon_indices[2] = indices[index_i / 3 + 2];
			fv[index_i / indices_in_tri] = polygon_indices;
		}

		PackedInt32Array bones = p_mesh[Mesh::ARRAY_BONES];
		Set<int32_t> bone_set;

		for (int32_t bones_i = 0; bones_i < bones.size(); bones_i++) {
			bone_set.insert(bones[bones_i]);
		}
		num_bones = bone_set.size();
		num_total_frames = 1;
		const int iteration_max = 100;
		double tolerance = 0.0;
		int patience = 3;
		DemBonesExt<_Scalar, _AniMeshScalar>::compute();
		double prevErr = -1;
		int np = 3;
		for (int32_t iteration = 0; iteration < iteration_max; iteration++) {
			double err = DemBones<_Scalar, _AniMeshScalar>::rmse();
			print_line("RMSE = " + itos(err));
			if ((err < prevErr * (1 + weightEps)) &&
					((prevErr - err) < tolerance * prevErr)) {
				np--;
				if (np == 0) {
					print_line("Convergence is reached!");
					return Array();
				}
			} else {
				np = patience;
			}
			prevErr = err;
			return Array();
		}
		return Array();
	}
};
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
