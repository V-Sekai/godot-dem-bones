#include <DemBones/DemBones.h>
#include <DemBones/DemBonesExt.h>
#include <DemBones/MatBlocks.h>

#include "core/config/engine.h"
#include "scene/resources/mesh.h"
#include "editor/import/scene_importer_mesh.h"


using namespace Dem;

class MyBones : DemBonesExt<double, float> {
// Turn EditorSceneImporterMesh with blend shapes into bones animations
public:
  Error dem_bones(Ref<EditorSceneImporterMesh> model) {
    const int iteration_max = 100;
    double tolerance = 0.0;
    int patience = 3;
    DemBonesExt<double, float>::compute();
    double prevErr = -1;
    int np = 3;
    for (int32_t iteration = 0; iteration < iteration_max; iteration++) {

      double err = rmse();
      print_line("RMSE = " + itos(err));
      if ((err < prevErr * (1 + weightEps)) &&
          ((prevErr - err) < tolerance * prevErr)) {
        np--;
        if (np == 0) {
          print_line("Convergence is reached!");
          return OK;
        }
      } else {

        np = patience;
      }
      prevErr = err;
      return FAILED;
    }
    return OK;
  }
};
