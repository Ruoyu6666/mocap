this file update: 2024_03_29


Experiment MOS1A
start: March 2023
Bogna Ignatowska-Jankowska, Hugo Hoedemaker, Lakshmipriya Swaminathan, Guo Da, Marylka Yoe Uusisaari
Okinawa Institute of Science and Technology


subjects
male c57BL mice
n=10
source: CLEA Japan
11-12 weeks old at the time of the first MoCap recording

DOB MC1 26-31_12_2022 n=3
DOB MC2 26-31_12_2022 n=3
DOB MC3 26-31_12_2022 n=4
arrived 22_02_2023

Implantation of GRIN lenses 4.0 mm 
covered with virus pAAV.syn.GCaMP6f.WPRE.SV40 #12 nRiM inventory 
to cerebellar nuclei -6.12AP, +/-2L/M, -2 D
02.03.2023 MC1
03.03.2023 MC2
06.03.2023 MC3


markers implanted: 
MC1 9.03.2023
MC2 10.03.2023
MC3 13.03.2023

phases of recordings:
MoCap only baseline: 23.03.2023
nVoke miniscope+Mocap dummies:28.03.2023
nVoke miniscope+MoCap baseline: 
MC2 31.03.2023
MC3 03.04.2023

nVoke miniscope + MoCap drug treatment (MoS1AD):
R1(s1-s7) 7.04.2023
R2 (s8-s13) 11.04.2023
R3 (s14-s19) 14.04.202
R4 (s20-s25) 18.04.2023
R5 (s26-s31) 21.04.2023
R6(s32-s37)25.04.2023
R7 (s38-s39)8.04.2023 *repetitions M9,M10
R8 (s40) 2.05.2023 *repetition M9

nVoke miniscope + MoCap drug treatment:
A vehicle
B CP55.940 0.3 mg/kg
C PF3845 30 mg/kg
D MJN110 2.5 mg/kg
E Harmaline 20 mg/kg (in double volume)

There are two vehicle groups 
A1 - 1 ml/100g (normal volume) 2 h pretreatment
A2- 2 ml/100g (double volume)30 min pretreatment

Administered i.p. at 1 ml/100 g of body mass except harmaline 2 ml/100g
Pretreatment time 2 hrs	for C and D, 30 mins for B and E
vehicle helf 30 min double volume, half normal volume 2 h pretreatment

Within subject Latin Square design								

There are x "samples" (e.g. "s7") 
7 individual mice x 5 treatments administered
tested in randomized order 
days 1,2,3,4,5 were spaced with at least 72 h wash out period

in rows 1-10 is INDIVIDUAL MOUSE:
in columns round/day

		7.04.2023	 11.04.2023		14.04.2023		18.04.2023		21.04.2023		25.04.2023		
M4_MC2	s1  A2		 s8		B		s14	C			s20	D			s26	E			s32	A1				
M5_MC2	s2	E		 s9		A1		s15	B			s21	C			s27	D			s33	A2				
M6_MC2	s3	D		 s10	E		s16	A1			s22	B			s28	C			s34	A2				
M7_MC3	s4	C	dead													
M8_MC3	s5	B		 s11	C		s17	D			s23	E			s29	A2			s35	A1				
M9_MC3	s6	A1		 s12	B		s18	C			s24	D			s30	E		s36	C*		s38	D*		s40	A2
M10_MC3	s7	E		 s13	A2		s19	B			s25	C			s31	D		s37	D*		s39	A1		s41	A1*


	TREATMENT				DOSE [mg/kg]	CONCENTRATION [mg/ml]		INJ VOLUME
A	VEHICLE 1:1:18, etOH:colliphor:saline							0			
B	CP55,940				0.3		0.03			normal volume	10 ml/kg
C	PF3845					30		3				normal volume	10 ml/kg
D	MJN110					2.5		0.25			normal volume	10 ml/kg
E	Harmaline				20		1				double volume	20 ml/ml
								
								
				
Filename example							
'MOS1aD_S38_M9_MC3_T3_TRM_2023_04_28_40MMIN'

MOS1aD=experiment
S38=sample chronological number
M9=individual mouse# 
MC3=cage id
T3=tail mark within cage
TRM=treadmil (various speeds in m/min)
CLB=climbing 
FL2=Floor2=OF=open_field 30 x 30 cm textured green floor (novel environment)
FL1=open field black floor for calibration
TRE=3D exploration (tree model)
2023_04_28=recording date
40MMIN=speed in case of treadmil
proc_bij_2024_01_20=processing date and initials who processed
C=treatment

sample number is chronological order in which animals were ran and it is a randomized order
to decode group asignment and individual animals it is best to use randomization table above and sort data accordingly

Recording
QTM 2022.2 build 7710 
300 Hz
60 s or 30 s in case of treadmill
disregard first 5s (1500 frames) and everything over 65 s
it takes 3 s to place animal in the arena +2 s slack

Markers
10 markers + miniscope

left_ankle
right_ankle
left_knee
right_knee
left_hip
right_hip
left_coord
right_coord
left_back
right_back
miniscope

"Hip" markets are directly over hip bone
"Coord" markers are approx 1 cm above hip bone (marker closest to center of mass)
"Back" markers are directly over shoulderblades
"Ankle" represents lower tibia bone near ankle joint
"Knee" represents higher tibia bone near knee joint
there may be a couple of recordings where back or coord is missing (mouse lost piercing during experiment)

markers are 6 mm (shoulder or coordinate),7 mm (hips), 8 mm (legs) long with 3 mm ball but 5 mm bals are used for experiments

inscopix 20 fps
