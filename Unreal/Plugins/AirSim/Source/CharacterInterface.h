// sena was here

#pragma once

#include "CharacterInterface.generated.h"

/* must have BlueprintType as a specifier to have this interface exposed to blueprints
 with this line you can easily add this interface to any blueprint class */
UINTERFACE(BlueprintType)
class UCharacterInterface : public UInterface
{
    GENERATED_UINTERFACE_BODY()
};

class ICharacterInterface
{
    GENERATED_IINTERFACE_BODY()
    
public:
    //classes using this interface must implement this
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "MyCategory")
    FVector getHumanPositionUpdated();
    
    //classes using this interface must implement this
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "MyCategory")
    TArray<FVector> getBonePositionsUpdated();
    
    //classes using this interface must implement this
    UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "MyCategory")
    void changeAnimation(int new_anim_num);
    
    UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "MyCategory")
    void changeCalibrationMode(bool calibMode);
};

