/**
 * @file joint_acceleration_constraint_unit.cpp
 * @brief TrajOpt IFOPT joint acceleration constraint unit tests
 *
 * @author Ben Greenberg
 * @date July 22, 2021
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2021, Southwest Research Institute
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <trajopt_utils/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <ctime>
#include <gtest/gtest.h>

#include <tesseract_environment/core/environment.h>
#include <tesseract_environment/ofkt/ofkt_state_solver.h>
#include <tesseract_kinematics/core/forward_kinematics.h>
#include <tesseract_environment/core/environment.h>
#include <tesseract_environment/core/utils.h>
#include <tesseract_common/types.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt_ifopt/variable_sets/joint_position_variable.h>
#include <trajopt_ifopt/constraints/joint_acceleration_constraint.h>
#include <trajopt_ifopt/utils/numeric_differentiation.h>
#include <trajopt_test_utils.hpp>

#include <console_bridge/console.h>

using namespace trajopt;
using namespace std;
using namespace util;
using namespace tesseract_environment;
using namespace tesseract_kinematics;
using namespace tesseract_collision;
using namespace tesseract_visualization;
using namespace tesseract_scene_graph;
using namespace tesseract_geometry;


class JointAccelerationConstraintUnit : public testing::TestWithParam<const char*>
{
public:
  Environment::Ptr env = std::make_shared<Environment>();
  ifopt::Problem nlp;

  tesseract_kinematics::ForwardKinematics::Ptr forward_kinematics;
  CartPosKinematicInfo::Ptr kinematic_info;
  CartPosConstraint::Ptr constraint;

  Eigen::Index n_dof;

  void SetUp() override
  {
    // Initialize Tesseract
    tesseract_common::fs::path urdf_file(std::string(TRAJOPT_DIR) + "/test/data/arm_around_table.urdf");
    tesseract_common::fs::path srdf_file(std::string(TRAJOPT_DIR) + "/test/data/pr2.srdf");
    tesseract_scene_graph::ResourceLocator::Ptr locator =
        std::make_shared<tesseract_scene_graph::SimpleResourceLocator>(locateResource);
    auto env = std::make_shared<Environment>();
    bool status = env->init<OFKTStateSolver>(urdf_file, srdf_file, locator);
    EXPECT_TRUE(status);

    // Extract necessary kinematic information
    forward_kinematics = env->getManipulatorManager()->getFwdKinematicSolver("right_arm");
    n_dof = forward_kinematics->numJoints();

    tesseract_environment::AdjacencyMap::Ptr adjacency_map = std::make_shared<tesseract_environment::AdjacencyMap>(
        env->getSceneGraph(), forward_kinematics->getActiveLinkNames(), env->getCurrentState()->link_transforms);
    kinematic_info = std::make_shared<trajopt::CartPosKinematicInfo>(
        forward_kinematics, adjacency_map, Eigen::Isometry3d::Identity(), forward_kinematics->getTipLinkName());

    auto pos = Eigen::VectorXd::Ones(forward_kinematics->numJoints());
    auto var0 = std::make_shared<trajopt::JointPosition>(pos, forward_kinematics->getJointNames(), "Joint_Position_0");
    nlp.AddVariableSet(var0);

    // 4) Add constraints
    // Add joint acceleration cost for all timesteps
    {
      Eigen::VectorXd accel_target = Eigen::VectorXd::Zero(6);
      auto accel_constraint = std::make_shared<trajopt::JointAccelConstraint>(accel_target, vars, "JointAcceleration");

      // Must link the variables to the constraint since that happens in AddConstraintSet
      accel_constraint->LinkWithVariables(nlp.GetOptVariables());
      auto accel_cost = std::make_shared<trajopt::SquaredCost>(accel_constraint, Eigen::VectorXd::Constant(accel_constraint->GetRows(), 1.0));
      nlp_.AddCostSet(accel_cost);
    }
  }
};


/**
 * @brief Tests the joint acceleration constraint
 */
TEST(JointAccelerationConstraintUnit, joint_position_1)  // NOLINT
{
  CONSOLE_BRIDGE_logDebug("JointAccelerationConstraintUnit, joint_position_1");

  // Run FK to get target pose
  Eigen::VectorXd joint_position = Eigen::VectorXd::Ones(n_dof);

  // Calculate jacobian numerically
  auto error_calculator = [&](const Eigen::Ref<const Eigen::VectorXd>& x) { return accel_cost->CalcValues(x); };
  trajopt::Jacobian num_jac_block = trajopt::calcForwardNumJac(error_calculator, joint_position, 1e-4);

  // Compare to constraint jacobian
  {
    trajopt::Jacobian jac_block(num_jac_block.rows(), num_jac_block.cols());
    accel_cost->CalcJacobianBlock(joint_position, jac_block);
    EXPECT_TRUE(jac_block.isApprox(num_jac_block, 1e-3));
    //      std::cout << "Numeric:\n" << num_jac_block.toDense() << std::endl;
    //      std::cout << "Analytic:\n" << jac_block.toDense() << std::endl;
  }
  {
    trajopt::Jacobian jac_block(num_jac_block.rows(), num_jac_block.cols());
    accel_cost->FillJacobianBlock("Joint_Acceleration_0", jac_block);
    EXPECT_TRUE(jac_block.toDense().isApprox(num_jac_block.toDense(), 1e-3));
    //      std::cout << "Numeric:\n" << num_jac_block.toDense() << std::endl;
    //      std::cout << "Analytic:\n" << jac_block.toDense() << std::endl;
  }
}

////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
