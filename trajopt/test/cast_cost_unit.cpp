#include <trajopt_utils/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <ctime>
#include <gtest/gtest.h>

#include <tesseract_collision/bullet/bullet_discrete_bvh_manager.h>
#include <tesseract_collision/bullet/bullet_cast_bvh_manager.h>
#include <tesseract_kinematics/kdl/kdl_fwd_kin_chain.h>
#include <tesseract_kinematics/kdl/kdl_fwd_kin_tree.h>
#include <tesseract_kinematics/core/utils.h>
#include <tesseract_environment/kdl/kdl_env.h>
#include <tesseract_environment/core/utils.h>
#include <tesseract_visualization/visualization.h>
#include <tesseract_scene_graph/graph.h>
#include <tesseract_scene_graph/parser/urdf_parser.h>
#include <tesseract_scene_graph/parser/srdf_parser.h>
#include <tesseract_scene_graph/utils.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt/collision_terms.hpp>
#include <trajopt/common.hpp>
#include <trajopt/plot_callback.hpp>
#include <trajopt/problem_description.hpp>
#include <trajopt_sco/optimizers.hpp>
#include <trajopt_test_utils.hpp>
#include <trajopt_utils/clock.hpp>
#include <trajopt_utils/config.hpp>
#include <trajopt_utils/eigen_conversions.hpp>
#include <trajopt_utils/logging.hpp>
#include <trajopt_utils/stl_to_string.hpp>

using namespace trajopt;
using namespace std;
using namespace util;
using namespace tesseract_environment;
using namespace tesseract_kinematics;
using namespace tesseract_collision;
using namespace tesseract_visualization;
using namespace tesseract_scene_graph;
using namespace tesseract_geometry;

static bool plotting = false;

class CastTest : public testing::TestWithParam<const char*>
{
public:
  SceneGraphPtr scene_graph_;             /**< Scene Graph */
  SRDFModel srdf_model_;                  /**< SRDF Model */
  KDLEnvPtr env_;                         /**< Trajopt Basic Environment */
  VisualizationPtr plotter_;              /**< Trajopt Plotter */
  ForwardKinematicsConstPtrMap kin_map_;  /**< A map between manipulator name and kinematics object */
//  tesseract_ros::ROSBasicPlottingPtr plotter_; /**< Trajopt Plotter */

  void SetUp() override
  {
    std::string urdf_file = std::string(TRAJOPT_DIR) + "/test/data/boxbot.urdf";
    std::string srdf_file = std::string(TRAJOPT_DIR) + "/test/data/boxbot.srdf";

    ResourceLocatorFn locator = locateResource;
    scene_graph_ = parseURDF(urdf_file, locator);
    EXPECT_TRUE(scene_graph_ != nullptr);
    EXPECT_TRUE(srdf_model_.initFile(*scene_graph_, srdf_file));

    // Add allowed collision to the scene
    processSRDFAllowedCollisions(*scene_graph_, srdf_model_);

    env_ = KDLEnvPtr(new KDLEnv);
    EXPECT_TRUE(env_ != nullptr);
    EXPECT_TRUE(env_->init(scene_graph_));

    // Register contact manager
    EXPECT_TRUE(env_->registerDiscreteContactManager("bullet", &tesseract_collision_bullet::BulletDiscreteBVHManager::create));
    EXPECT_TRUE(env_->registerContinuousContactManager("bullet", &tesseract_collision_bullet::BulletCastBVHManager::create));

    // Set Active contact manager
    EXPECT_TRUE(env_->setActiveDiscreteContactManager("bullet"));
    EXPECT_TRUE(env_->setActiveContinuousContactManager("bullet"));

    // Generate Kinematics Map
    kin_map_ = createKinematicsMap<KDLFwdKinChain, KDLFwdKinTree>(scene_graph_, srdf_model_);

    gLogLevel = util::LevelError;

    // Create plotting tool
//    plotter_.reset(new tesseract_ros::ROSBasicPlotting(env_));
  }
};

TEST_F(CastTest, boxes)
{
  CONSOLE_BRIDGE_logDebug("CastTest, boxes");

  Json::Value root = readJsonFile(std::string(TRAJOPT_DIR)  + "/test/data/config/box_cast_test.json");

  std::unordered_map<std::string, double> ipos;
  ipos["boxbot_x_joint"] = -1.9;
  ipos["boxbot_y_joint"] = 0;
  env_->setState(ipos);

//  plotter_->plotScene();

  TrajOptProbPtr prob = ConstructProblem(root, env_, kin_map_);
  ASSERT_TRUE(!!prob);

  std::vector<ContactResultMap> collisions;
  ContinuousContactManagerPtr manager = prob->GetEnv()->getContinuousContactManager();
  AdjacencyMapPtr adjacency_map = std::make_shared<AdjacencyMap>(scene_graph_,
                                                                 prob->GetKin()->getActiveLinkNames(),
                                                                 prob->GetEnv()->getState()->transforms);

  manager->setActiveCollisionObjects(adjacency_map->getActiveLinkNames());
  manager->setContactDistanceThreshold(0);

  collisions.clear();
  bool found = checkTrajectory(*manager, *prob->GetEnv(), prob->GetKin()->getJointNames(), prob->GetInitTraj(), collisions);

  EXPECT_TRUE(found);
  CONSOLE_BRIDGE_logDebug((found) ? ("Initial trajectory is in collision") : ("Initial trajectory is collision free"));

  sco::BasicTrustRegionSQP opt(prob);
  if (plotting)
    opt.addCallback(PlotCallback(*prob, plotter_));
  opt.initialize(trajToDblVec(prob->GetInitTraj()));
  opt.optimize();

  if (plotting)
    plotter_->clear();

  collisions.clear();
  found = checkTrajectory(*manager, *prob->GetEnv(), prob->GetKin()->getJointNames(), getTraj(opt.x(), prob->GetVars()), collisions);

  EXPECT_FALSE(found);
  CONSOLE_BRIDGE_logDebug((found) ? ("Final trajectory is in collision") : ("Final trajectory is collision free"));
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);

//  pnh.param("plotting", plotting, false);
  return RUN_ALL_TESTS();
}
