// sena was here

#pragma once

#include "DroneInterface.generated.h"

/* must have BlueprintType as a specifier to have this interface exposed to blueprints
 with this line you can easily add this interface to any blueprint class */
UINTERFACE(BlueprintType)
class UDroneInterface : public UInterface
{
    GENERATED_UINTERFACE_BODY()
};

class IDroneInterface
{
    GENERATED_IINTERFACE_BODY()
    
public:
    //classes using this interface must implement this
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "MyCategory")
    FVector getDronePositionUpdated();
    
    //classes using this interface must implement this
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "MyCategory")
    FRotator getDroneOrientationUpdated();
    
};
