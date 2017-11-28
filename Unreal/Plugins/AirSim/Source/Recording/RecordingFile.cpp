#include "RecordingFile.h"
#include "HAL/PlatformFilemanager.h"
#include "FileHelper.h"
#include <sstream>
#include "ImageUtils.h"
#include "common/ClockFactory.hpp"
#include "common/common_utils/FileSystem.hpp"


RecordingFile::RecordingFile(std::vector <std::string> columns)
{
    this->columns = columns;
}
void RecordingFile::appendRecord(TArray<uint8>& image_data, VehiclePawnWrapper* wrapper, const msr::airlib::Vector3r_arr bonesPos)
{
    if (image_data.Num() == 0)
    return;
    
    bool imageSavedOk = false;
    FString filePath;
    
    std::string filename = std::string("img_").append(std::to_string(images_saved_)).append(".png");
    
    try {
        filePath = FString(common_utils::FileSystem::combine(image_path_, filename).c_str());
        //sena was here
        writeString(createLogLine(bonesPos).append(filename).append("\n"));
        imageSavedOk = FFileHelper::SaveArrayToFile(image_data, *filePath);
    }
    catch(std::exception& ex) {
        UAirBlueprintLib::LogMessage(TEXT("Image file save failed"), FString(ex.what()), LogDebugLevel::Failure);
    }
    // If render command is complete, save image along with position and orientation
    
    if (imageSavedOk) {
        //writeString(wrapper->getLogLine().append(filename).append("\n")); //sena was here
        
        UAirBlueprintLib::LogMessage(TEXT("Screenshot saved to:"), filePath, LogDebugLevel::Success);
        images_saved_++;
    }
}

//sena was here
std::string RecordingFile::createLogLine(msr::airlib::Vector3r_arr bones)
{
    uint64_t timestamp_millis = static_cast<uint64_t>(msr::airlib::ClockFactory::get()->nowNanos() / 1.0E6);
    
    //TODO: because this bug we are using alternative code with stringstream
    //https://answers.unrealengine.com/questions/664905/unreal-crashes-on-two-lines-of-extremely-simple-st.html
    std::string line;
    line.append(std::to_string(timestamp_millis)).append("\t")
    .append(std::to_string(bones.dronePos.x())).append("\t")
    .append(std::to_string(bones.dronePos.y())).append("\t")
    .append(std::to_string(bones.dronePos.z())).append("\t")
    .append(std::to_string(bones.droneOrient.x())).append("\t")
    .append(std::to_string(bones.droneOrient.y())).append("\t")
    .append(std::to_string(bones.droneOrient.z())).append("\t")
    .append(std::to_string(bones.humanPos.x())).append("\t")
    .append(std::to_string(bones.humanPos.y())).append("\t")
    .append(std::to_string(bones.humanPos.z())).append("\t")
    .append(std::to_string(bones.hip.x())).append("\t")
    .append(std::to_string(bones.hip.y())).append("\t")
    .append(std::to_string(bones.hip.z())).append("\t")
    .append(std::to_string(bones.right_up_leg.x())).append("\t")
    .append(std::to_string(bones.right_up_leg.y())).append("\t")
    .append(std::to_string(bones.right_up_leg.z())).append("\t")
    .append(std::to_string(bones.right_leg.x())).append("\t")
    .append(std::to_string(bones.right_leg.y())).append("\t")
    .append(std::to_string(bones.right_leg.z())).append("\t")
    .append(std::to_string(bones.right_foot.x())).append("\t")
    .append(std::to_string(bones.right_foot.y())).append("\t")
    .append(std::to_string(bones.right_foot.z())).append("\t")
    .append(std::to_string(bones.left_up_leg.x())).append("\t")
    .append(std::to_string(bones.left_up_leg.y())).append("\t")
    .append(std::to_string(bones.left_up_leg.z())).append("\t")
    .append(std::to_string(bones.left_leg.x())).append("\t")
    .append(std::to_string(bones.left_leg.y())).append("\t")
    .append(std::to_string(bones.left_leg.z())).append("\t")
    .append(std::to_string(bones.left_foot.x())).append("\t")
    .append(std::to_string(bones.left_foot.y())).append("\t")
    .append(std::to_string(bones.left_foot.z())).append("\t")
    .append(std::to_string(bones.spine1.x())).append("\t")
    .append(std::to_string(bones.spine1.y())).append("\t")
    .append(std::to_string(bones.spine1.z())).append("\t")
    .append(std::to_string(bones.neck.x())).append("\t")
    .append(std::to_string(bones.neck.y())).append("\t")
    .append(std::to_string(bones.neck.z())).append("\t")
    .append(std::to_string(bones.head.x())).append("\t")
    .append(std::to_string(bones.head.y())).append("\t")
    .append(std::to_string(bones.head.z())).append("\t")
    .append(std::to_string(bones.left_arm.x())).append("\t")
    .append(std::to_string(bones.left_arm.y())).append("\t")
    .append(std::to_string(bones.left_arm.z())).append("\t")
    .append(std::to_string(bones.left_forearm.x())).append("\t")
    .append(std::to_string(bones.left_forearm.y())).append("\t")
    .append(std::to_string(bones.left_forearm.z())).append("\t")
    .append(std::to_string(bones.left_hand.x())).append("\t")
    .append(std::to_string(bones.left_hand.y())).append("\t")
    .append(std::to_string(bones.left_hand.z())).append("\t")
    .append(std::to_string(bones.right_arm.x())).append("\t")
    .append(std::to_string(bones.right_arm.y())).append("\t")
    .append(std::to_string(bones.right_arm.z())).append("\t")
    .append(std::to_string(bones.right_forearm.x())).append("\t")
    .append(std::to_string(bones.right_forearm.y())).append("\t")
    .append(std::to_string(bones.right_forearm.z())).append("\t")
    .append(std::to_string(bones.right_hand.x())).append("\t")
    .append(std::to_string(bones.right_hand.y())).append("\t")
    .append(std::to_string(bones.right_hand.z())).append("\t")
    .append(std::to_string(bones.right_hand_tip.x())).append("\t")
    .append(std::to_string(bones.right_hand_tip.y())).append("\t")
    .append(std::to_string(bones.right_hand_tip.z())).append("\t")
    .append(std::to_string(bones.left_hand_tip.x())).append("\t")
    .append(std::to_string(bones.left_hand_tip.y())).append("\t")
    .append(std::to_string(bones.left_hand_tip.z())).append("\t")
    .append(std::to_string(bones.right_foot_tip.x())).append("\t")
    .append(std::to_string(bones.right_foot_tip.y())).append("\t")
    .append(std::to_string(bones.right_foot_tip.z())).append("\t")
    .append(std::to_string(bones.left_foot_tip.x())).append("\t")
    .append(std::to_string(bones.left_foot_tip.y())).append("\t")
    .append(std::to_string(bones.left_foot_tip.z())).append("\t");
    return line;
}

void RecordingFile::appendColumnHeader(std::vector <std::string> columns)
{
    std::string line;
    for (int i = 0; i < columns.size()-1; i++)
    {
        line.append(columns[i]).append("\t");
    }
    line.append(columns[columns.size() - 1]).append("\n");
    
    writeString(line);
}

void RecordingFile::createFile(const std::string& file_path)
{
    try {
        closeFile();
        
        IPlatformFile& platform_file = FPlatformFileManager::Get().GetPlatformFile();
        log_file_handle_ = platform_file.OpenWrite(*FString(file_path.c_str()));
        appendColumnHeader(this->columns);
    }
    catch(std::exception& ex) {
        UAirBlueprintLib::LogMessageString(std::string("createFile Failed for ") + file_path, ex.what(), LogDebugLevel::Failure);
    }
}

bool RecordingFile::isFileOpen()
{
    return log_file_handle_ != nullptr;
}

void RecordingFile::closeFile()
{
    if (isFileOpen())
    delete log_file_handle_;
    
    log_file_handle_ = nullptr;
}

void RecordingFile::writeString(const std::string& str)
{
    try {
        if (log_file_handle_) {
            FString line_f = FString(str.c_str());
            log_file_handle_->Write((const uint8*)TCHAR_TO_ANSI(*line_f), line_f.Len());
        }
        else
        UAirBlueprintLib::LogMessageString("Attempt to write to recording log file when file was not opened", "", LogDebugLevel::Failure);
    }
    catch(std::exception& ex) {
        UAirBlueprintLib::LogMessageString(std::string("file write to recording file failed "), ex.what(), LogDebugLevel::Failure);
    }
}

RecordingFile::~RecordingFile()
{
    stopRecording(true);
}

void RecordingFile::startRecording()
{
    try {
        std::string log_folderpath = common_utils::FileSystem::getLogFolderPath(true);
        image_path_ = common_utils::FileSystem::ensureFolder(log_folderpath, "images");
        std::string log_filepath = common_utils::FileSystem::getLogFileNamePath(log_folderpath, record_filename, "", ".txt", false);
        if (log_filepath != "")
        createFile(log_filepath);
        else {
            UAirBlueprintLib::LogMessageString("Cannot start recording because path for log file is not available", "", LogDebugLevel::Failure);
            return;
        }
        
        if (isFileOpen()) {
            is_recording_ = true;
            
            UAirBlueprintLib::LogMessage(TEXT("Recording: "), TEXT("Started"), LogDebugLevel::Success);
        }
        else
        UAirBlueprintLib::LogMessageString("Error creating log file", log_filepath.c_str(), LogDebugLevel::Failure);
    }
    catch(...) {
        UAirBlueprintLib::LogMessageString("Error in startRecording", "", LogDebugLevel::Failure);
    }
}

void RecordingFile::stopRecording(bool ignore_if_stopped)
{
    is_recording_ = false;
    if (! isFileOpen()) {
        if (ignore_if_stopped)
        return;
        
        UAirBlueprintLib::LogMessage(TEXT("Recording Error"), TEXT("File was not open"), LogDebugLevel::Failure);
    }
    else
    closeFile();
    
    UAirBlueprintLib::LogMessage(TEXT("Recording: "), TEXT("Stopped"), LogDebugLevel::Success);
    UAirBlueprintLib::LogMessage(TEXT("Data saved to: "), FString(image_path_.c_str()), LogDebugLevel::Success);
}

bool RecordingFile::isRecording()
{
    return is_recording_;
}

