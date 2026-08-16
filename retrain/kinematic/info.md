# Mocap
- All joints: 
    left_ankle,  left_back,  left_coord,  left_hip,  left_knee, right_ankle, right_back, right_coord, right_hip, right_knee

# Sdannce
head (Snout, EarL, EarR, SpineF) \
trunk (SpineM, SpineL, TailBase, ShoulderL, ShoulderR, HipL, HipR) \
forelimbs (ElbowL, WristL, HandL,ElbowR, WristR, HandR) \
forelimbs (KneeL, AnkleL, FootL, KneeR, AnkleR, FootR)

joints_sdannce = ["Snout",  "EarL", "EarR",  "SpineF", "SpineM", "SpineL", "TailBase", \
                  "ShoulderL", "ElbowL", "WristL", "HandL", "ShoulderR", "ElbowR", "WristR", "HandR", \ 
                  "HipL", "KneeL", "AnkleL", "FootL", "HipR", "KneeR", "AnkleR", "FootR",]




# Features
- Posture / body configuration: 
    1. **Trunk length**: shoulder mid - hip mid
    2. **Body height**: vertical distance from body center to menan ankle
    3. Whole body height
    4. **Trunk inclination**: angle of trunk vector(hip_mid → shoulder_mid) relative to the vertical axis
    5. **Trunk orientation** / heading direction: (angle between the shoulder–hip line and the ground)
    6. **Orientation/heading angular velocity**: frame-to-frame change of heading direction
    - Elongation ratio: body_length / body_width

- Upper body movement
    7. **body bend angle/torso curvature**: (coord_mid → shoulder_mid) and (coord_mid → hip_mid) mid-point/per side
    - (body bend angle difference)
    - (hip width, back width, shoulder width, shoulder height difference, hip height difference, ankle height difference)
        
- Leg/limb configuration
    7. **Hip Angles**: between (hip→shoulder/coord) and (hip→knee) per side
    9. **Knee Angles**: between (hip→knee) and (knee→ankle) per side
    - Knee angle difference
    - Stance width (left right ankle distance)
    - (hip to ankle vertical distance)
    - Ankle speed (subtract centroid velocity)
    - (leg extension by length)

- Locomotion: 
    11. **Speed (centroid)** 
    12. **Acceleration  (centroid)**
    - (whole body motion energy)
    - (joint velocity, average, max)
    

- Position away from the beginning
