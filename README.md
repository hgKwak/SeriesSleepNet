# SeriesSleepNet
Minji Lee, Hyeong-Jin Kim, Heon-Gyu Kwak, Dong-Ok Won, and Seong-Whan Lee, "SeriesSleepNet: An EEG Time Series Model Using Temporal CNN with Bi-LSTM for Automatic Sleep Stage Scoring*.

<img src="/img/fig1.png" width="100%" height="100%"></img>

## How to run
1. Download the Sleep-EDF and SHHS database
   + Sleep-EDF database is available [here][sleep-edf].   
      **Note that, we used only 20 people of the Sleep Cassette data.**    
      **_i.e.)_ from subject SC4001 to SC4192 in the sleep-cassette.**    
      
   + SHHS database is available [here][SHHS]. 
   
[sleep-edf]: https://physionet.org/content/sleep-edfx/1.0.0/
[SHHS]: https://sleepdata.org/datasets/shhs/

2. Data pre-processing
   + Change directory to ```./SerieseSleepNet/pre-processing```
   + Place downloaded dataset folders in ```./SerieseSleepNet/pre-processing/data```
  
      * _i.e.)_ ```./SerieseSleepNet/pre-processing/data/shhs``` or ```./SerieseSleepNet/pre-processing/data/sleep-edf-database-expanded-1.0.0```
   + Unzip ```eeglab.zip```
   + Run preprocessing files 
      
      * ```sleepedf_preprocessing.m```, ```shhs_preprocessing.m```, ```male_shhs_preprocessing.m```, ```female_shhs_preprocessing.m```
   
3. Training and testing the SerieseSleepNet
   + Change directory to ```./SerieseSleepNet```
   + Refer to ```example.txt``` to train and test SerieseSleepNet.
   
## Environment:
+ Matlab R2019b
+ Python 3.8
+ Pytorch v1.70
+ numpy v1.19.5
+ scipy v1.5.0
+ scikit-learn v 0.24.2
+ tqdm v4.59.0

## Result:
Hypnogram and posterior probability distribution with SerieseSleepNet for one subject of Sleep-EDF database

## Contact:
+ Minji Lee (Department of Brain and Cognitive Engineering, Korea University; minjilee@korea.ac.kr)
+ Hyeong-Jin Kim (Department of Brain and Cognitive Engineering, Korea University; kme0115@korea.ac.kr)
+ Heon-Gyu Kwak (Department of Artificial Intelligence, Korea University; hg_kwak@korea.ac.kr)

## License:
This project is released under the MIT License.
