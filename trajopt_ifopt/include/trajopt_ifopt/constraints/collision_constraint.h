/**
 * @file collision_constraint.h
 * @brief The collision position constraint
 *
 * @author Levi Armstrong
 * @author Matthew Powelson
 * @date May 18, 2020
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2020, Southwest Research Institute
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
#ifndef TRAJOPT_IFOPT_COLLISION_CONSTRAINT_H
#define TRAJOPT_IFOPT_COLLISION_CONSTRAINT_H

#include <trajopt_utils/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <Eigen/Eigen>
#include <ifopt/constraint_set.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt_ifopt/variable_sets/joint_position_variable.h>
#include <trajopt_ifopt/utils/cache.h>

namespace trajopt
{
/**
 * @brief This contains the different types of expression evaluators used when performing continuous collision checking.
 */
enum class CollisionEvaluatorType
{
  START_FREE_END_FREE = 0,  /**< @brief Both start and end state variables are free to be adjusted */
  START_FREE_END_FIXED = 1, /**< @brief Only start state variables are free to be adjusted */
  START_FIXED_END_FREE = 2, /**< @brief Only end state variables are free to be adjusted */

  /**
   * @brief Both start and end state variables are free to be adjusted
   * The jacobian is calculated using a weighted sum over the interpolated states for a given link pair.
   */
  START_FREE_END_FREE_WEIGHTED_SUM = 3,
  /**
   * @brief Only start state variables are free to be adjusted
   * The jacobian is calculated using a weighted sum over the interpolated states for a given link pair.
   */
  START_FREE_END_FIXED_WEIGHTED_SUM = 4,
  /**
   * @brief Only end state variables are free to be adjusted
   * The jacobian is calculated using a weighted sum over the interpolated states for a given link pair.
   */
  START_FIXED_END_FREE_WEIGHTED_SUM = 5,

  SINGLE_TIME_STEP = 6, /**< @brief Expressions are only calculated at a single time step */
  /**
   * @brief Expressions are only calculated at a single time step
   * The jacobian is calculated using a weighted sum for a given link pair.
   */
  SINGLE_TIME_STEP_WEIGHTED_SUM = 7
};

class CollisionConstraintIfopt : public ifopt::ConstraintSet
{
public:
  using Ptr = std::shared_ptr<CollisionConstraintIfopt>;
  using ConstPtr = std::shared_ptr<const CollisionConstraintIfopt>;

  CollisionConstraintIfopt(SingleTimestepCollisionEvaluator::Ptr collision_evaluator,
                           JointPosition::ConstPtr position_var,
                           const std::string& name = "Collision");

  CollisionConstraintIfopt(tesseract_kinematics::ForwardKinematics::ConstPtr manip,
                           tesseract_environment::Environment::ConstPtr env,
                           tesseract_environment::AdjacencyMap::ConstPtr adjacency_map,
                           const Eigen::Isometry3d& world_to_base,
                           SafetyMarginData::ConstPtr safety_margin_data,
                           tesseract_collision::ContactTestType contact_test_type,
                           double longest_valid_segment_length,
                           JointPosition::ConstPtr position_var,
                           CollisionExpressionEvaluatorType type,
                           double safety_margin_buffer,
                           const std::string& name = "Collision",
                           bool dynamic_environment = false);


  /** @brief Calculates the values associated with the constraint */
  Eigen::VectorXd CalcValues(const Eigen::Ref<Eigen::VectorXd>& joint_vals) const;

  /**
   * @brief Returns the values associated with the constraint.
   * @return
   */
  Eigen::VectorXd GetValues() const override;

  /**
   * @brief  Returns the "bounds" of this constraint. How these are enforced is up to the solver
   * @return Returns the "bounds" of this constraint
   */
  std::vector<ifopt::Bounds> GetBounds() const override;

  /**
   * @brief Sets the bounds on the collision distance
   * @param bounds New bounds that will be set. Should be size 1
   */
  void SetBounds(const std::vector<ifopt::Bounds>& bounds);

  /**
   * @brief Fills the jacobian block associated with the constraint
   * @param jac_block Block of the overall jacobian associated with these constraints
   */
  void CalcJacobianBlock(const Eigen::Ref<Eigen::VectorXd>& joint_vals, Jacobian& jac_block) const;
  /**
   * @brief Fills the jacobian block associated with the given var_set.
   * @param var_set Name of the var_set to which the jac_block is associated
   * @param jac_block Block of the overall jacobian associated with these constraints and the var_set variable
   */
  void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override;

  /**
   * @brief Given optimizer parameters calculate the collision results for this evaluator
   * @param x Optimizer variables
   * @param dist_results Contact results map
   */
  void CalcCollisions(const Eigen::Ref<Eigen::VectorXd>& joint_vals, tesseract_collision::ContactResultMap& dist_results);

  /**
   * @brief Calculate the collision results for this evaluator using the current variable set
   * @param dist_map Contact results map
   * @param dist_vector Contact results vector
   */
  void CalcCollisions(tesseract_collision::ContactResultMap& dist_map,
                      tesseract_collision::ContactResultVector& dist_vector);

  /**
   * @brief This function checks to see if results are cached for variable set. If not it calls CalcCollisions and
   * caches the results vector with x as the key.
   * @param dist_vector Contact results vector
   */
  void GetCollisionsCached(tesseract_collision::ContactResultVector& dist_vector);

  /**
   * @brief This function checks to see if results are cached for input variable x. If not it calls CalcCollisions and
   * caches the results with x as the key.
   *  @param dist_map Contact results map
   */
  void GetCollisionsCached(tesseract_collision::ContactResultMap& dist_map);

private:
  /** @brief The number of joints in a single JointPosition */
  long n_dof_;

  /** @brief Bounds on the constraint value. Default: std::vector<Bounds>(1, ifopt::BoundSmallerZero) */
  std::vector<ifopt::Bounds> bounds_;

  /** @brief Pointers to the vars used by this constraint.
   *
   * Do not access them directly. Instead use this->GetVariables()->GetComponent(position_var->GetName())->GetValues()*/
  JointPosition::ConstPtr position_var_;

  tesseract_kinematics::ForwardKinematics::ConstPtr manip_;
  tesseract_environment::Environment::ConstPtr env_;
  tesseract_environment::AdjacencyMap::ConstPtr adjacency_map_;
  Eigen::Isometry3d world_to_base_;
  SafetyMarginData::ConstPtr safety_margin_data_;
  double safety_margin_buffer_;
  tesseract_collision::ContactTestType contact_test_type_;
  double longest_valid_segment_length_;
  tesseract_environment::StateSolver::Ptr state_solver_;
  CollisionEvaluatorType evaluator_type_;
  std::function<tesseract_environment::EnvState::Ptr(const std::vector<std::string>& joint_names,
                                                     const Eigen::Ref<const Eigen::VectorXd>& joint_values)>
      get_state_fn_;
  bool dynamic_environment_;

  /** @brief Cache the collision results */
  Cache<size_t, std::pair<tesseract_collision::ContactResultMap, tesseract_collision::ContactResultVector>, 10> cache_;
};
};  // namespace trajopt
#endif
