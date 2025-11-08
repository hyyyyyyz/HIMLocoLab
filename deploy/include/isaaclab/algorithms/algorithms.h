// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <mutex>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::vector<float> obs) = 0;
    
    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        Ort::TypeInfo input_type = session->GetInputTypeInfo(0);
        input_shape = input_type.GetTensorTypeAndShapeInfo().GetShape();
        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();

        action.resize(output_shape[1]);
    }

    std::vector<float> act(std::vector<float> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, obs.data(), obs.size(), input_shape.data(), input_shape.size());
        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
        auto floatarr = output_tensor.front().GetTensorMutableData<float>();

        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    const std::vector<const char*> input_names = {"obs"};
    const std::vector<const char*> output_names = {"actions"};

    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
};

class DualOrtRunner : public Algorithms
{
public:
    DualOrtRunner(std::string encoder_path, std::string policy_path)
    {
        // Init ONNX environment
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_dual_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        // Load encoder network
        encoder_session = std::make_unique<Ort::Session>(env, encoder_path.c_str(), session_options);

        Ort::TypeInfo encoder_input_type = encoder_session->GetInputTypeInfo(0);
        encoder_input_shape = encoder_input_type.GetTensorTypeAndShapeInfo().GetShape();
        Ort::TypeInfo encoder_output_type = encoder_session->GetOutputTypeInfo(0);
        encoder_output_shape = encoder_output_type.GetTensorTypeAndShapeInfo().GetShape();

        // Load policy network
        policy_session = std::make_unique<Ort::Session>(env, policy_path.c_str(), session_options);

        Ort::TypeInfo policy_input_type = policy_session->GetInputTypeInfo(0);
        policy_input_shape = policy_input_type.GetTensorTypeAndShapeInfo().GetShape();
        Ort::TypeInfo policy_output_type = policy_session->GetOutputTypeInfo(0);
        policy_output_shape = policy_output_type.GetTensorTypeAndShapeInfo().GetShape();


        // Resize buffers
        encoder_output.resize(encoder_output_shape[1]);
        action.resize(policy_output_shape[1]);
        
        // policy_input = [current_obs + encoder_output]
        current_obs_dim = policy_input_shape[1] - encoder_output_shape[1];
    }    
    
    std::vector<float> act(std::vector<float> obs)
    {
        size_t encoder_input_size = encoder_input_shape[1];

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // Step 1: Run encoder on obs_history
        std::vector<float> obs_history(obs.begin(), obs.begin() + encoder_input_size);
        
        auto encoder_input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            obs_history.data(), 
            obs_history.size(), 
            encoder_input_shape.data(), 
            encoder_input_shape.size()
        );
        auto encoder_output_tensor = encoder_session->Run(
            Ort::RunOptions{nullptr}, 
            encoder_input_names.data(), 
            &encoder_input_tensor, 
            1, 
            encoder_output_names.data(), 
            1
        );
        
        auto encoder_floatarr = encoder_output_tensor.front().GetTensorMutableData<float>();
        std::memcpy(encoder_output.data(), encoder_floatarr, encoder_output_shape[1] * sizeof(float));

        // Step 2: Concatenate current_obs + encoder_output for policy
        std::vector<float> policy_input;
        policy_input.insert(policy_input.end(), obs.begin(), obs.begin() + current_obs_dim);  // current_obs (dynamic)
        policy_input.insert(policy_input.end(), encoder_output.begin(), encoder_output.end());  // encoder output

        auto policy_input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            policy_input.data(), 
            policy_input.size(), 
            policy_input_shape.data(), 
            policy_input_shape.size()
        );

        // Step 3: Run policy
        auto policy_output_tensor = policy_session->Run(
            Ort::RunOptions{nullptr}, 
            policy_input_names.data(), 
            &policy_input_tensor, 
            1, 
            policy_output_names.data(), 
            1
        );

        auto policy_floatarr = policy_output_tensor.front().GetTensorMutableData<float>();

        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), policy_floatarr, policy_output_shape[1] * sizeof(float));
        
        return action;
    }

    std::vector<float> get_encoder_output()
    {
        return encoder_output;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    
    // Encoder network
    std::unique_ptr<Ort::Session> encoder_session;
    std::vector<int64_t> encoder_input_shape;
    std::vector<int64_t> encoder_output_shape;
    size_t current_obs_dim;
    std::vector<float> encoder_output;
    const std::vector<const char*> encoder_input_names = {"obs_history"};
    const std::vector<const char*> encoder_output_names = {"encoder_output"};
    
    // Policy network
    std::unique_ptr<Ort::Session> policy_session;
    std::vector<int64_t> policy_input_shape;
    std::vector<int64_t> policy_output_shape;
    const std::vector<const char*> policy_input_names = {"obs"};
    const std::vector<const char*> policy_output_names = {"actions"};
};

};