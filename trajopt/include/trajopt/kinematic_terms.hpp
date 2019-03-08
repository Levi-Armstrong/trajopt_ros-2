#pragma once
#include <trajopt_utils/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <Eigen/Core>

#include <tesseract_environment/core/environment.h>
#include <tesseract_environment/core/utils.h>
#include <tesseract_kinematics/core/forward_kinematics.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt/common.hpp>
#include <trajopt_sco/modeling.hpp>
#include <trajopt_sco/modeling_utils.hpp>
#include <trajopt_sco/sco_fwd.hpp>

namespace trajopt
{
/**
 * @brief Used to calculate the error for CartPoseTermInfo
 * This is converted to a cost or constraint using TrajOptCostFromErrFunc or TrajOptConstraintFromErrFunc
 */
struct DynamicCartPoseErrCalculator : public TrajOptVectorOfVector
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::string target_;
  std::pair<std::string, Eigen::Isometry3d> kin_target_;
  tesseract_kinematics::ForwardKinematicsConstPtr manip_;
  tesseract_environment::AdjacencyMapConstPtr adjacency_map_;
  Eigen::Isometry3d world_to_base_;
  std::string link_;
  std::pair<std::string, Eigen::Isometry3d> kin_link_;
  Eigen::Isometry3d tcp_;
  DynamicCartPoseErrCalculator(const std::string& target,
                               tesseract_kinematics::ForwardKinematicsConstPtr manip,
                               tesseract_environment::AdjacencyMapConstPtr adjacency_map,
                               Eigen::Isometry3d world_to_base,
                               std::string link,
                               Eigen::Isometry3d tcp = Eigen::Isometry3d::Identity())
    : target_(target), manip_(manip), adjacency_map_(adjacency_map), world_to_base_(world_to_base), link_(link), tcp_(tcp)
  {
    kin_link_ = adjacency_map_->at(link_);
    kin_target_ = adjacency_map_->at(target_);
  }

  void Plot(const tesseract_visualization::VisualizationPtr& plotter, const Eigen::VectorXd& dof_vals) override;

  Eigen::VectorXd operator()(const Eigen::VectorXd& dof_vals) const override;

};

/**
 * @brief Used to calculate the error for StaticCartPoseTermInfo
 * This is converted to a cost or constraint using TrajOptCostFromErrFunc or TrajOptConstraintFromErrFunc
 */
struct CartPoseErrCalculator : public TrajOptVectorOfVector
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Isometry3d pose_inv_;
  tesseract_kinematics::ForwardKinematicsConstPtr manip_;
  tesseract_environment::AdjacencyMapConstPtr adjacency_map_;
  Eigen::Isometry3d world_to_base_;
  std::string link_;
  std::pair<std::string, Eigen::Isometry3d> kin_link_;
  Eigen::Isometry3d tcp_;
  CartPoseErrCalculator(const Eigen::Isometry3d& pose,
                        tesseract_kinematics::ForwardKinematicsConstPtr manip,
                        tesseract_environment::AdjacencyMapConstPtr adjacency_map,
                        Eigen::Isometry3d world_to_base,
                        std::string link,
                        Eigen::Isometry3d tcp = Eigen::Isometry3d::Identity())
    : pose_inv_(pose.inverse()), manip_(manip), adjacency_map_(adjacency_map), world_to_base_(world_to_base), link_(link), tcp_(tcp)
  {
    kin_link_ = adjacency_map_->at(link_);
  }

  void Plot(const tesseract_visualization::VisualizationPtr& plotter, const Eigen::VectorXd& dof_vals) override;

  Eigen::VectorXd operator()(const Eigen::VectorXd& dof_vals) const override;
};

/**
 * @brief Used to calculate the jacobian for CartVelTermInfo
 *
 */
struct CartVelJacCalculator : sco::MatrixOfVector
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  tesseract_kinematics::ForwardKinematicsConstPtr manip_;
  tesseract_environment::AdjacencyMapConstPtr adjacency_map_;
  Eigen::Isometry3d world_to_base_;
  std::string link_;
  std::pair<std::string, Eigen::Isometry3d> kin_link_;
  double limit_;
  Eigen::Isometry3d tcp_;
  CartVelJacCalculator(tesseract_kinematics::ForwardKinematicsConstPtr manip,
                       tesseract_environment::AdjacencyMapConstPtr adjacency_map,
                       Eigen::Isometry3d world_to_base,
                       std::string link,
                       double limit,
                       Eigen::Isometry3d tcp = Eigen::Isometry3d::Identity())
    : manip_(manip), adjacency_map_(adjacency_map), world_to_base_(world_to_base), link_(link), limit_(limit), tcp_(tcp)
  {
    kin_link_ = adjacency_map_->at(link_);
  }

  Eigen::MatrixXd operator()(const Eigen::VectorXd& dof_vals) const override;
};

/**
 * @brief  Used to calculate the error for CartVelTermInfo
 * This is converted to a cost or constraint using TrajOptCostFromErrFunc or TrajOptConstraintFromErrFunc
 */
struct CartVelErrCalculator : sco::VectorOfVector
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  tesseract_kinematics::ForwardKinematicsConstPtr manip_;
  tesseract_environment::AdjacencyMapConstPtr adjacency_map_;
  Eigen::Isometry3d world_to_base_;
  std::string link_;
  std::pair<std::string, Eigen::Isometry3d> kin_link_;
  double limit_;
  Eigen::Isometry3d tcp_;
  CartVelErrCalculator(tesseract_kinematics::ForwardKinematicsConstPtr manip,
                       tesseract_environment::AdjacencyMapConstPtr adjacency_map,
                       Eigen::Isometry3d world_to_base,
                       std::string link,
                       double limit,
                       Eigen::Isometry3d tcp = Eigen::Isometry3d::Identity())
    : manip_(manip), adjacency_map_(adjacency_map), world_to_base_(world_to_base), link_(link), limit_(limit), tcp_(tcp)
  {
    kin_link_ = adjacency_map_->at(link_);
  }

  Eigen::VectorXd operator()(const Eigen::VectorXd& dof_vals) const override;
};
}
