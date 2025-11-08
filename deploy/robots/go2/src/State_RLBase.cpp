#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    spdlog::info("Initializing State_{}...", state_string);
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = parser_policy_dir(cfg["policy_dir"].as<std::string>());

    // Load deploy configuration
    auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    
    // Check if dual network mode is enabled (from deploy.yaml)
    bool use_encoder = deploy_cfg["use_encoder"] ? deploy_cfg["use_encoder"].as<bool>() : false;
    
    if (use_encoder) {
        spdlog::info("Loading dual network (encoder + policy)...");
        env->alg = std::make_unique<isaaclab::DualOrtRunner>(
            (policy_dir / "exported" / "encoder.onnx").string(),
            (policy_dir / "exported" / "policy.onnx").string()
        );
    } else {
        spdlog::info("Loading single network (policy only)...");
        env->alg = std::make_unique<isaaclab::OrtRunner>(
            (policy_dir / "exported" / "policy.onnx").string()
        );
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            (int)FSMMode::Passive
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}