/**
 * @file collision_constraint.cpp
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
#include <trajopt_ifopt/constraints/collision_constraint.h>
#include <trajopt/collision_terms.hpp>

TRAJOPT_IGNORE_WARNINGS_PUSH
#include <tesseract_kinematics/core/utils.h>
#include <console_bridge/console.h>
TRAJOPT_IGNORE_WARNINGS_POP

namespace trajopt
{
CollisionConstraintIfopt::CollisionConstraintIfopt(SingleTimestepCollisionEvaluator::Ptr collision_evaluator,
                                                   JointPosition::ConstPtr position_var,
                                                   const std::string& name)
  : ifopt::ConstraintSet(1, name)
  , position_var_(std::move(position_var))
  , collision_evaluator_(std::move(collision_evaluator))
{
  // Set n_dof_ for convenience
  n_dof_ = position_var_->GetRows();
  assert(n_dof_ > 0);

  bounds_ = std::vector<ifopt::Bounds>(1, ifopt::BoundSmallerZero);
}

CollisionConstraintIfopt::CollisionConstraintIfopt(tesseract_kinematics::ForwardKinematics::ConstPtr manip,
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
                                                   bool dynamic_environment)
  : ifopt::ConstraintSet(1, name)
  , position_var_(std::move(position_var))
  , manip_(std::move(manip))
  , env_(std::move(env))
  , adjacency_map_(std::move(adjacency_map))
  , world_to_base_(world_to_base)
  , safety_margin_data_(std::move(safety_margin_data))
  , safety_margin_buffer_(safety_margin_buffer)
  , contact_test_type_(contact_test_type)
  , longest_valid_segment_length_(longest_valid_segment_length)
  , state_solver_(env_->getStateSolver())
  , dynamic_environment_(dynamic_environment)
{
  // If the environment is not expected to change, then the cloned state solver may be used each time.
  if (dynamic_environment_)
    get_state_fn_ = [&](const std::vector<std::string>& joint_names,
                        const Eigen::Ref<const Eigen::VectorXd>& joint_values) {
      return env_->getState(joint_names, joint_values);
    };
  else
    get_state_fn_ = [&](const std::vector<std::string>& joint_names,
                        const Eigen::Ref<const Eigen::VectorXd>& joint_values) {
      return state_solver_->getState(joint_names, joint_values);
    };
}

Eigen::VectorXd CollisionConstraintIfopt::CalcValues(const Eigen::Ref<Eigen::VectorXd>& joint_vals) const
{
  Eigen::VectorXd err = Eigen::VectorXd::Zero(1);

  // Check the collisions
  tesseract_collision::ContactResultVector dist_results;
  {
    tesseract_collision::ContactResultMap dist_results_map;
    collision_evaluator_->CalcCollisions(joint_vals, dist_results_map);
    tesseract_collision::flattenMoveResults(std::move(dist_results_map), dist_results);
  }

  for (tesseract_collision::ContactResult& dist_result : dist_results)
  {
    // Contains the contact distance threshold and coefficient for the given link pair
    const Eigen::Vector2d& data = safety_margin_data_->getPairSafetyMarginData(
        dist_result.link_names[0], dist_result.link_names[1]);
    // distance will be distance from threshold with negative being greater (further) than the threshold times the
    // coeff
    err[0] += sco::pospart((data[0] - dist_result.distance) * data[1]);
  }
  return err;
}

Eigen::VectorXd CollisionConstraintIfopt::GetValues() const
{
  // Get current joint values
  Eigen::VectorXd joint_vals = this->GetVariables()->GetComponent(position_var_->GetName())->GetValues();

  return CalcValues(joint_vals);
}

// Set the limits on the constraint values
std::vector<ifopt::Bounds> CollisionConstraintIfopt::GetBounds() const { return bounds_; }

void CollisionConstraintIfopt::SetBounds(const std::vector<ifopt::Bounds>& bounds)
{
  assert(bounds.size() == 1);
  bounds_ = bounds;
}

void CollisionConstraintIfopt::CalcJacobianBlock(const Eigen::Ref<Eigen::VectorXd>& joint_vals,
                                                 Jacobian& jac_block) const
{
  // Reserve enough room in the sparse matrix
  jac_block.reserve(n_dof_);

  // Calculate collisions
  tesseract_collision::ContactResultVector dist_results;
  {
    tesseract_collision::ContactResultMap dist_results_map;
    CalcCollisions(joint_vals, dist_results_map);
    tesseract_collision::flattenMoveResults(std::move(dist_results_map), dist_results);
  }

  // Get gradients for all contacts
  std::vector<trajopt::GradientResults> grad_results;
  grad_results.reserve(dist_results.size());
  for (tesseract_collision::ContactResult& dist_result : dist_results)
  {
    // Contains the contact distance threshold and coefficient for the given link pair
    const Eigen::Vector2d& data = safety_margin_data_->getPairSafetyMarginData(
        dist_result.link_names[0], dist_result.link_names[1]);
    grad_results.push_back(collision_evaluator_->GetGradient(joint_vals, dist_result, data, true));
  }

  // Convert GradientResults to jacobian
  int idx = 0;
  Eigen::VectorXd grad_vec = Eigen::VectorXd::Zero(n_dof_);
  for (auto& grad : grad_results)
  {
    if (grad.gradients[0].has_gradient)
      grad_vec += grad.gradients[0].gradient;
    if (grad.gradients[1].has_gradient)
      grad_vec += grad.gradients[1].gradient;
    idx++;
  }

  // This does work but could be faster
  for (int j = 0; j < n_dof_; j++)
  {
    // Collision is 1 x n_dof
    jac_block.coeffRef(0, j) = -1 * grad_vec[j];
  }
}

void CollisionConstraintIfopt::FillJacobianBlock(std::string var_set, Jacobian& jac_block) const
{
  // Only modify the jacobian if this constraint uses var_set
  if (var_set == position_var_->GetName())
  {
    // Get current joint values
    VectorXd joint_vals = this->GetVariables()->GetComponent(position_var_->GetName())->GetValues();

    CalcJacobianBlock(joint_vals, jac_block);
  }
}

void CollisionConstraintIfopt::CalcCollisions(const Eigen::Ref<Eigen::VectorXd> &joint_vals,
                                              tesseract_collision::ContactResultMap& dist_results)
{
  tesseract_environment::EnvState::Ptr state = get_state_fn_(manip_->getJointNames(), joint_vals);

  for (const auto& link_name : env_->getActiveLinkNames())
    contact_manager_->setCollisionObjectsTransform(link_name, state->link_transforms[link_name]);

  contact_manager_->contactTest(dist_results, contact_test_type_);

  for (auto& pair : dist_results)
  {
    // Contains the contact distance threshold and coefficient for the given link pair
    const Eigen::Vector2d& data = safety_margin_data_->getPairSafetyMarginData(pair.first.first, pair.first.second);
    auto end = std::remove_if(
        pair.second.begin(), pair.second.end(), [&data, this](const tesseract_collision::ContactResult& r) {
          return (!((data[0] + safety_margin_buffer_) > r.distance));
        });
    pair.second.erase(end, pair.second.end());
  }
}

void CollisionConstraintIfopt::CalcCollisions(tesseract_collision::ContactResultMap& dist_map,
                                              tesseract_collision::ContactResultVector& dist_vector)
{
  CalcCollisions(GetVariables()->GetComponent(position_var_->GetName())->GetValues(), dist_map);
  tesseract_collision::flattenCopyResults(dist_map, dist_vector);
}


inline size_t hash(const Eigen::VectorXd& x) { return boost::hash_range(&x(0), &x(0) + x.size()); }
void CollisionConstraintIfopt::GetCollisionsCached(tesseract_collision::ContactResultVector& dist_vector)
{
  size_t key = hash(GetVariables()->GetComponent(position_var_->GetName())->GetValues());
  auto it = cache_.get(key);
  if (it != nullptr)
  {
    CONSOLE_BRIDGE_logDebug("Uing cached collision check");
    dist_vector = it->second;
  }
  else
  {
    CONSOLE_BRIDGE_logDebug("Not using cached collision check");
    tesseract_collision::ContactResultMap dist_map;
    CalcCollisions(dist_map, dist_vector);
    cache_.put(key, std::make_pair(dist_map, dist_vector));
  }
}

void CollisionConstraintIfopt::GetCollisionsCached(tesseract_collision::ContactResultMap& dist_map)
{
  size_t key = hash(GetVariables()->GetComponent(position_var_->GetName())->GetValues());
  auto it = cache_.get(key);
  if (it != nullptr)
  {
    CONSOLE_BRIDGE_logDebug("Uing cached collision check");
    dist_map = it->first;
  }
  else
  {
    CONSOLE_BRIDGE_logDebug("Not using cached collision check");
    tesseract_collision::ContactResultVector dist_vector;
    CalcCollisions(dist_map, dist_vector);
    cache_.put(key, std::make_pair(dist_map, dist_vector));
  }
}

}  // namespace trajopt
