# EDER-VIANN
Emulsion Data Event Reconstruction - Vertex Identification by using Artificial Neural Networks

This README just serves as a very short user guide, the documentation will be written much later

Installation steps
--

1) pip3 install tensorflow==1.14.0 --user
2) pip3 install keras==2.3.1 --user
3) pip3 install tensorflow-gpu==1.14.0 --user
   Only required if you have intent to create/train CNN models
4) go to your home directory on AFS where you would like to install the package
5) git clone https://github.com/FilipsFedotovs/EDER-VIANN/
6) cd EDER-VIANN/
7) python3 setup.py
8) The installation will require another directory, please enter the location on EOS where you would like to keep data and the models
   Has to provide up to 10-100 GB of storage depending on whether particular components of the framework is used. An example of the input is /eos/user/<username      first letter>/<user name> . In theory AFS work location also can be specified but it is not recommended.
9) The installer will copy and analyse existing data and the pre-trained model, it might take 5-10 minutes.
10) if the message 'EDER-VIANN setup is successfully completed' is displayed, it means that the package is ready for work




Vertex Reconstruction
--
1) Please make sure that you have a file with hits that there were reconstructed as Tracks.
   Following columns are required: 
   - Track ID Quadrant or Event ID (if MC truth track reconstruction data is used)
   - Track ID (FEDRA or MC depending which data is used)
   - x-coordinates of the track hits
   - y-coordinates of the track hits
   - z-coordinates of the track hits

2) Please open $AFS/EDER_VIANN/Code/Utilities/Parameters.py and check that the lines between 6-13 
   Within the list of naming conventions correspond to headers in the file that you intend to use.

3) Check the 'CNN_Model_Name' variable - it has the name of the Model that is used for reconstruction (included in the package). If you wish to use your own,        please place it in the $EOS/EDER_VIANN/Models and change the 'CNN_Model_Name' variable accordingly. You might need to change resolution and MaxX, MaxY, MaxZ      parameters if the model was trained with images that have had different size because the model might fail.

4) If happy, save and close the file.

5) cd ..

6) tmux 
   please note the number of lxplus machine at which tmux session is logged in

7) kinit your<username>@CERN.CH -l 24h00m

8) python3 R1_PrepareRecData.py --Xmin 50000 --Xmax 60000 --Ymin 50000 --Ymax 60000 --Track FEDRA --f $<your file with reconstructed tracks> 
   min and max value arguments can be changed or completely removed if all ECC data to be reconstructed. Track type can be changed to MC if Monte-Carlo truth        track reconstruction data is used. The script can take 1-5 minutes depending on the size of the input file. Once it finish it will give the message "The track    data has been created successfully and written to ....' and exit.

9) python3 R2_GenerateSeeds.py --Mode R
   The script will send warning, type Y. The program will send HTCondor jobs and exit. The jobs take about an hour.

10) python3 R2_GenerateSeeds.py --Mode C
   It will check whether the HTCondor jobs have been completed, if not it will give a warning.
   If the jobs are completed it will remove duplicates from the seeds and generate the following message: "Seed generation is completed".

11) python3 R3_FilterSeeds.py --Mode R
    The script will send warning, type Y. The program will send HTCondor jobs and exit. The jobs can take few hours.

12) python3 python3 R3_FilterSeeds.py --Mode C 
    It will check whether the HTCondor jobs have been completed, if not it will give a warning.

13) python3 R4_VertexSeeds.py --Mode R 
    The script will send warning, type Y. The program will send HTCondor jobs and exit. The jobs can take few hours.

14) python3 R4_VertexSeeds.py --Mode C 
   It will check whether the HTCondor jobs have been completed, if not it will give a warning.The output will produce a file R4_REC_SEEDS.csv with a list of seeds with each one given a probability value. 

15) python3 R5_MergeVertices.py 
    The execution can take up to several hours if the data size is big. The program will produce the R5_REC_VERTICES file. In the file each line will contain a       list of tracks and the vertex number. The script can be ran with the option '--Acceptance' which takes in account only the seeds with probability above the       given value (has to be between 0 and 1)
   
   
   
EDER-VIANN Vertex Reconstruction Evaluation
--
Can only be used if there is a data available with MC vertex truth information.
1) python3 E1_PrepareEvalData.py --Xmin 50000 --Xmax 60000 --Ymin 50000 --Ymax 60000 --Track FEDRA --f $<your file with reconstructed tracks> (min and max value      arguments have to match those that were used in for previous phase in Step 7).The script can take 1-5 minutes depending on the size of the input file. Once it    finish it will give the message "The track data has been created successfully and written to ....' and exit.)
2) python3 E2_GenerateEvalSeeds.py --Mode C (The script will send warning, type Y. The program will send HTCondor jobs and exit. The jobs take about an hour.)
3) (After an hour or so) python3 E2_GenerateEvalSeeds.py --Mode C (it will check whether the HTCondor jobs have been completed, if not it will give a warning).
   If the jobs are completed it will remove duplicates from the seeds and generate the following message: "Seed generation is completed".
   python3 E3_DecorateEvalSeeds.py --Mode R (The script will send warning, type Y. The program will send HTCondor jobs and exit. The jobs take about an hour.)
   (After an hour or so) python3 E2_GenerateEvalSeeds.py --Mode C (it will check whether the HTCondor jobs have been completed, if not it will give a warning).
   -------- FEDRA Vertex Reconstruction Evaluation -----
   --Can only be used if there is a data available with FEDRA vertex reconstruction information.
1) python3 E7_PrepareKalmanData.py --Xmin 50000 --Xmax 60000 --Ymin 50000 --Ymax 60000 --f $<your file with reconstructed tracks>
2) python3 E8_GenerateKalmanSeeds.py --Mode R (The script will send warning, type Y. The program will send HTCondor jobs and exit. The jobs will take about an hour.)
3) (After an hour or so) python3 E8_GenerateKalmanSeeds.py --Mode C (it will check whether the HTCondor jobs have been completed, if not it will give a warning).
4) E9_DecorateKalmanSeeds.py --Mode R (The script will send warning, type Y. The program will send HTCondor jobs and exit. The jobs will take about an hour.)
   (After an hour or so) E9_DecorateKalmanSeeds.py --Mode C (it will check whether the HTCondor jobs have been completed, if not it will give a warning).
5) python3 E10_EvaluateKalmanData.py (The script will return the perecision and the recall of the FEDRA reconstruction output)
   
-------- EDER-VIANN Model Training ------
1)  python3 M1_PrepareTrainData.py --Xmin 50000 --Xmax 120000 --Ymin -120000 --Ymax 50000 --Track FEDRA  --f $<your file with reconstructed tracks> (min and max       value arguments can be changed or completely removed if all ECC data to be reconstructed. The X and Y bounds are exclusive (they define the portion of the ECC     data that is not used in training). Track type can be changed to MC if Monte-Carlo truth track reconstruction data is used. The script can take 1-5 minutes       depending on the size of the input file. Once it finish it will give the message "The track data has been created successfully and written to ....' and exit.)
2)  python3 M2_GenerateTrainSeeds.py --Mode R (The script will send warning, type Y. The program will send HTCondor jobs and exit. The jobs take about an hour.)
3) (After an hour or so) M2_GenerateTrainSeeds.py --Mode C (it will check whether the HTCondor jobs have been completed, if not it will give a warning).
  
