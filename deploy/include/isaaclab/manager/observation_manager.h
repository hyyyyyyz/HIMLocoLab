// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <unordered_set>
#include <deque>
#include "isaaclab/manager/manager_term_cfg.h"
#include <iostream>
#include <spdlog/spdlog.h>

namespace isaaclab
{

using ObsMap = std::map<std::string, ObsFunc>;

inline ObsMap& observations_map() {
    static ObsMap instance;
    return instance;
}

#define REGISTER_OBSERVATION(name) \
    inline std::vector<float> name(ManagerBasedRLEnv* env); \
    inline struct name##_registrar { \
        name##_registrar() { observations_map()[#name] = name; } \
    } name##_registrar_instance; \
    inline std::vector<float> name(ManagerBasedRLEnv* env)


class ObservationManager
{
public:
    ObservationManager(YAML::Node cfg, ManagerBasedRLEnv* env)
    :cfg(cfg), env(env)
    {
        // Read global history length
        history_length = cfg["history_length"] ? cfg["history_length"].as<int>() : 1;
        _prapare_terms();
        _initialize_history_buffer();
    }

    void reset()
    {
        for(auto & term : obs_term_cfgs)
        {
            term.reset(term.func(this->env));
        }
        // Clear and reinitialize history buffer with zeros
        obs_history_buffer.clear();
        // Pre-fill buffer with zero observations so first step gets 6-step history with zeros
        std::vector<float> zero_obs(get_obs_dim(), 0.0f);
        for (int i = 0; i < history_length; ++i) {
            obs_history_buffer.push_back(zero_obs);
        }
    }

    std::vector<float> compute()
    {
        // Step 1: Compute current observations (without history in individual terms)
        std::vector<float> current_obs;
        for(auto & term : obs_term_cfgs)
        {
            auto term_obs = term.func(this->env);  // Get raw observation
            
            // Apply scaling
            if (term.scale.size() > 0) {
                for (size_t i = 0; i < term_obs.size(); ++i) {
                    term_obs[i] *= term.scale[i];
                }
            }
            
            // Apply clipping
            if (term.clip.size() == 2) {
                for (auto& val : term_obs) {
                    val = std::max(term.clip[0], std::min(val, term.clip[1]));
                }
            }
            
            current_obs.insert(current_obs.end(), term_obs.begin(), term_obs.end());
        }
        
        
        // Step 2: Add current observation to history buffer with sliding window
        if (obs_history_buffer.size() >= (size_t)history_length) {
            obs_history_buffer.pop_front();
        }
        obs_history_buffer.push_back(current_obs);
        
        // Step 3: Stack observations - order: [current, history1, history2, ..., historyN]
        // This matches the training wrapper stacking order
        std::vector<float> stacked_obs;
        stacked_obs.insert(stacked_obs.end(), current_obs.begin(), current_obs.end());
        
        // Add historical observations (oldest to newest before current)
        std::vector<std::vector<float>> history_stack(obs_history_buffer.begin(), obs_history_buffer.end());
        if (history_stack.size() > 1) {
            // Skip the last one (current obs) and add the rest
            for (size_t i = 0; i < history_stack.size() - 1; ++i) {
                stacked_obs.insert(stacked_obs.end(), history_stack[i].begin(), history_stack[i].end());
            }
        }
        
        
        return stacked_obs;
    }

    YAML::Node cfg;
    ManagerBasedRLEnv* env;
    int history_length;
    std::deque<std::vector<float>> obs_history_buffer;
    
    // Get observation dimension (sum of all term dimensions)
    size_t get_obs_dim() const
    {
        size_t total_dim = 0;
        for (const auto& term_cfg : obs_term_cfgs) {
            auto sample_obs = term_cfg.func(const_cast<ManagerBasedRLEnv*>(env));
            total_dim += sample_obs.size();
        }
        return total_dim;
    }
    
private:
    void _initialize_history_buffer()
    {
        // Get the observation dimension from terms
        size_t obs_dim = 0;
        for (const auto& term_cfg : obs_term_cfgs) {
            auto sample_obs = term_cfg.func(env);
            obs_dim += sample_obs.size();
        }
        
        // Pre-fill buffer with zero observations
        std::vector<float> zero_obs(obs_dim, 0.0f);
        obs_history_buffer.clear();
        for (int i = 0; i < history_length; ++i) {
            obs_history_buffer.push_back(zero_obs);
        }

        spdlog::info("History buffer initialized with {} zero observations (dim={})", history_length, get_obs_dim());
    }

    void _prapare_terms()
    {
        // Get observations config from the full config
        auto obs_cfg = cfg["observations"];
        
        if (!obs_cfg) {
            throw std::runtime_error("No 'observations' key found in config");
        }
        
        for(auto it = obs_cfg.begin(); it != obs_cfg.end(); ++it)
        {
            auto term_yaml_cfg = it->second;
            ObservationTermCfg term_cfg;
            // Don't set history_length for individual terms - we handle it globally
            term_cfg.history_length = 1;

            auto term_name = it->first.as<std::string>();
            if(observations_map()[term_name] == nullptr)
            {
                throw std::runtime_error("Observation term '" + term_name + "' is not registered.");
            }
            term_cfg.func = observations_map()[term_name];   

            auto obs = term_cfg.func(this->env);
            term_cfg.reset(obs);
            term_cfg.scale = term_yaml_cfg["scale"].as<std::vector<float>>();
            if(!term_yaml_cfg["clip"].IsNull()) {
                term_cfg.clip = term_yaml_cfg["clip"].as<std::vector<float>>();
            }

            this->obs_term_cfgs.push_back(term_cfg);
            spdlog::info("Successfully loaded observation term: {}", term_name);
            
        }
    }
    
    std::vector<ObservationTermCfg> obs_term_cfgs;
};

};