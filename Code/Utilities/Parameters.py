#This is the list of parameters that EDER-VIANN uses for reconstruction, model training etc. There have been collated here in one place for the user convenience
# Part of EDER-VIANN package
#Made by Filips Fedotovs
#Current version 1.0



Config='SND'
if Config=='SHIP':
    ######List of naming conventions
    x='x' #Column name x-coordinate of the track hit
    y='y' #Column name for y-coordinate of the track hit
    z='z' #Column name for z-coordinate of the track hit
    FEDRA_Track_ID='FEDRATrackID' #Column nameActual track id for FEDRA (or other reconstruction software)
    FEDRA_Track_QUADRANT='quarter' #Quarter of the ECC where the track is reconstructed If not present in the data please put the Track ID (the same as above)
    MC_Track_ID='MCTrack'  #Column name for Track ID for MC Truth reconstruction data
    MC_Event_ID='MCEvent' #Column name for Event id for MC truth reconstruction data (If absent please enter the MCTrack as for above)
    MC_VX_ID='MotherID'   #Column name for Track mother id (For MC truth only)
    FEDRA_VX_ID='VertexS' #Column name for the reconstructed Vertex id
    MC_NV_VX_ID='-1'
    FEDRA_NV_VX_ID='-1.0'
    ########List of the package run parameters
    MaxTracksPerJob=20000 #This parameter imposes the limit on the number of the tracks form the Start plate when forming the Seeds.
    MaxEvalTracksPerJob=20000 #This parameter imposes the limit on the number of the tracks form the Start plate when forming the Seeds.
    MaxSeedsPerJob=40000
    MaxVxPerJob=10000
    MaxSeedsPerVxPool=20000

    ######List of geometrical constain parameters
    SI_1=1200
    SI_2=1310
    SI_3=1600
    SI_4=2620
    SI_5=3120
    SI_6=3940
    SI_7=4000 #This parameter restricts the maximum euclidean distance between the first hits of the 2-track seeds that are subject to the Vertex Fit.
    MinHitsTrack=2
    MaxTrainSampleSize=50000
    MaxValSampleSize=100000
    VO_T=3900 #The minimum distance from the reconstructed Vertex Origin to the closest starting hit of any track in the seed
    VO_max_Z=437925 #Fidu
    VO_min_Z=360340
    MaxDoca=200
    MinAngle=0 #Seed Opening Angle (Magnitude) in radians
    MaxAngle=4 #Seed Opening Angle (Magnitude) in radians



    ##Model parameters
    acceptance=0.5
    pre_vx_acceptance=0.662
    link_acceptance=1.2
    resolution=100
    MaxX=100.0
    MaxY=100.0
    MaxZ=1315
    GNNMaxX=100.0
    GNNMaxY=100.0
    GNNMaxZ=1315
    CNN_Model_Name='2T_100_FEDRA_2_model'


    def Seed_Bond_Fit_Acceptance(row):
        if row['AntiLink_Strenth']>0:
          return 1.16*(row['Link_Strength']+row['Seed_CNN_Fit'])/row['AntiLink_Strenth']
        else:
          return 100

    ModelArchitecture=\
        [[4, 4, 1, 2, 2, 2, 2], #Layer 1
            [5, 4, 1, 1, 2, 2, 2], #Layer 2
            [5, 4, 2, 2, 2, 2, 2], #Layer 3
            [], #Layer 4
            [], #Layer 5
            [6, 4, 2], #Dense Layer 1
            [5, 4, 2], #Dense Layer 2
            [4, 4, 2], #Dense Layer 3
            [], #Dense Layer 4
            [], #Dense Layer 5
            [7, 1, 1, 4]] #Output Layer
if Config=='SND':
    ######List of naming conventions
    x='x' #Column name x-coordinate of the track hit
    y='y' #Column name for y-coordinate of the track hit
    z='z' #Column name for z-coordinate of the track hit
    FEDRA_Track_ID='FEDRA Track ID' #Column nameActual track id for FEDRA (or other reconstruction software)
    FEDRA_Track_QUADRANT='Quadrant' #Quarter of the ECC where the track is reconstructed If not present in the data please put the Track ID (the same as above)
    MC_Track_ID='MC Track'  #Column name for Track ID for MC Truth reconstruction data
    MC_Event_ID='MC Event' #Column name for Event id for MC truth reconstruction data (If absent please enter the MCTrack as for above)
    MC_VX_ID='MC Mother ID'   #Column name for Track mother id (For MC truth only)
    MC_VX_PDG='Mother PDG'   #Column name for Track mother id (For MC truth only)
    FEDRA_VX_ID='VertexS' #Column name for the reconstructed Vertex id
    MC_NV_VX_ID=['-2','-1']
    FEDRA_NV_VX_ID=['-1.0']
    MC_SGNL_VX_PDG=[]
    ########List of the package run parameters
    MaxTracksPerJob=20000 #This parameter imposes the limit on the number of the tracks form the Start plate when forming the Seeds.
    MaxEvalTracksPerJob=20000 #This parameter imposes the limit on the number of the tracks form the Start plate when forming the Seeds.
    MaxSeedsPerJob=40000
    MaxVxPerJob=10000
    MaxSeedsPerVxPool=20000

    ######List of geometrical constain parameters
    SI_1=1200
    SI_2=1310
    SI_3=1600
    SI_4=2620
    SI_5=3120
    SI_6=3940
    SI_7=4000 #This parameter restricts the maximum euclidean distance between the first hits of the 2-track seeds that are subject to the Vertex Fit.
    MinHitsTrack=2
    MaxTrainSampleSize=50000
    MaxValSampleSize=100000
    VO_T=3900 #The minimum distance from the reconstructed Vertex Origin to the closest starting hit of any track in the seed
    VO_max_Z=437925 #Fidu
    VO_min_Z=360340
    MaxDoca=200
    MinAngle=0 #Seed Opening Angle (Magnitude) in radians
    MaxAngle=4 #Seed Opening Angle (Magnitude) in radians



    ##Model parameters
    acceptance=0.5
    pre_vx_acceptance=0.662
    link_acceptance=1.2
    resolution=50
    MaxX=10000
    MaxY=10000
    MaxZ=3000
    CNN_Model_Name='2T_100_FEDRA_2_model'


    def Seed_Bond_Fit_Acceptance(row):
        if row['AntiLink_Strenth']>0:
          return 1.16*(row['Link_Strength']+row['Seed_CNN_Fit'])/row['AntiLink_Strenth']
        else:
          return 100

    ModelArchitecture=\
        [[1, 4, 1, 2, 2, 2, 2], #Layer 1
            [], #Layer 2
            [], #Layer 3
            [], #Layer 4
            [], #Layer 5
            [1, 4, 2], #Dense Layer 1
            [], #Dense Layer 2
            [], #Dense Layer 3
            [], #Dense Layer 4
            [], #Dense Layer 5
            [7, 1, 1, 4]] #Output Layer
            
#    ModelArchitecture=\
#        [[4, 4, 1, 2, 2, 2, 2], #Layer 1
#            [5, 4, 1, 1, 2, 2, 2], #Layer 2
#            [5, 4, 2, 2, 2, 2, 2], #Layer 3
#            [], #Layer 4
#            [], #Layer 5
#            [6, 4, 2], #Dense Layer 1
#            [5, 4, 2], #Dense Layer 2
#            [4, 4, 2], #Dense Layer 3
#            [], #Dense Layer 4
#            [], #Dense Layer 5
#            [7, 1, 1, 4]] #Output Layer

