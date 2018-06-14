#pragma once

#include "CoreMinimal.h"
#include "Engine/TextureRenderTarget2D.h"

#include <memory>
#include "common/common_utils/WorkerThread.hpp"
#include "common/CommonStructs.hpp"


class RenderRequest : public FRenderCommand
{
public:
    typedef msr::airlib::Vector3r_arr Vector3r_arr;
    
public:
    UTexture2D *Texture2D; //sena was here

    struct RenderParams {
        UTextureRenderTarget2D* render_target;
        bool pixels_as_float;
        bool compress;
        
        RenderParams(UTextureRenderTarget2D* render_target_val, bool pixels_as_float_val, bool compress_val)
        : render_target(render_target_val), pixels_as_float(pixels_as_float_val), compress(compress_val)
        {
        }
    };
    struct RenderResult {
        TArray<uint8> image_data_uint8;
        TArray<float> image_data_float;
        
        TArray<FColor> bmp;
        TArray<FFloat16Color> bmp_float;
        
        int width;
        int height;
        Vector3r_arr bonePos_data; //sena was here
        msr::airlib::TTimePoint time_stamp;
    };
    
private:
    static FReadSurfaceDataFlags setupRenderResource(const FTextureRenderTargetResource* rt_resource, const RenderParams* params, RenderResult* result, FIntPoint& size);
    bool use_safe_method_;
    Vector3r_arr* bonesPosPtr; //sena was here
    std::shared_ptr<RenderParams> *params_;
    std::shared_ptr<RenderResult> *results_;
    unsigned int req_size_;
    
    std::shared_ptr<msr::airlib::WorkerThreadSignal> wait_signal_;
    
    
public:
    RenderRequest(bool use_safe_method = false, Vector3r_arr* bonesPosPtr_ = nullptr);
    ~RenderRequest();
    
    void DoTask(ENamedThreads::Type CurrentThread, const FGraphEventRef& MyCompletionGraphEvent)
    {
        ExecuteTask();
    }
    
    FORCEINLINE TStatId GetStatId() const
    {
        RETURN_QUICK_DECLARE_CYCLE_STAT(RenderRequest, STATGROUP_RenderThreadCommands);
    }
    
    // read pixels from render target using render thread, then compress the result into PNG
    // argument on the thread that calls this method.
    
    void getScreenshot(std::shared_ptr<RenderParams> params[], std::vector<std::shared_ptr<RenderResult>>& results, unsigned int req_size);
    void SimpleGetScreenshot(UTextureRenderTarget2D* textureTarget, std::vector<std::shared_ptr<RenderResult>>&  results); //sena was here
    
    void ExecuteTask();
};

