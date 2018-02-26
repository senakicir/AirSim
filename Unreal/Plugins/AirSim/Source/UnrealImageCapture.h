#pragma once

#include "CoreMinimal.h"
#include "PIPCamera.h"
#include "common/ImageCaptureBase.hpp"


class UnrealImageCapture : public msr::airlib::ImageCaptureBase
{
public:
    typedef msr::airlib::ImageCaptureBase::ImageType ImageType;
    typedef msr::airlib::Vector3r_arr Vector3r_arr; //sena was here

    UnrealImageCapture(const std::vector<APIPCamera*>& cameras);
    virtual ~UnrealImageCapture();

    //sena was here
    virtual void getImages(const std::vector<ImageRequest>& requests, std::vector<ImageResponse>& responses, Vector3r_arr* bonePosPtr) override;

private:
    //sena was here
    void getSceneCaptureImage(const std::vector<msr::airlib::ImageCaptureBase::ImageRequest>& requests, std::vector<msr::airlib::ImageCaptureBase::ImageResponse>& responses, bool use_safe_method, Vector3r_arr* bonePosPtr);
    
    void addScreenCaptureHandler(UWorld *world);
    bool getScreenshotScreen(ImageType image_type, std::vector<uint8_t>& compressedPng);
    
    void updateCameraVisibility(APIPCamera* camera, const msr::airlib::ImageCaptureBase::ImageRequest& request);
private:
    std::vector<APIPCamera*> cameras_;
    std::vector<uint8_t> last_compressed_png_;
};


