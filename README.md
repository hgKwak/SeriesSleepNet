# SeriesSleepNet
Minji Lee, Heon-Gyu Kwak, Hyeong-Jin Kim, Dong-Ok Won, Ki-Young Jung, and Seong-Whan Lee, "SeriesSleepNet: An EEG Time Series Model with
Partial Data Augmentation Using Single-Channel
EEG for Automatic Sleep Stage Scoring*.

<img src="/img/architecture.png" width="100%" height="100%"></img>

## How to run
1. Download the Sleep-EDF and SHHS database
   + Sleep-EDF database is available [here][sleep-edf].   
      **Note that, we used only 20 people of the Sleep Cassette data.**     
   + SHHS database is available [here][SHHS]. 
   
[sleep-edf]: https://physionet.org/content/sleep-edfx/1.0.0/
[SHHS]: https://sleepdata.org/datasets/shhs/

2. Data pre-processing
    + Place downloaded datasets in ```./SeriesSleepNet/data/raw/``` and change the ```data_dir``` argument value in ```prepare_physionet.py``` and ```prepare_shhs.py``` files, according to your local settings.    
    + Run preprocessing files.
	** _i.e.)_ ```prepare_physionet.py (for Sleep-EDF dataset)```, ```prepare_shhs.py (for SHHS dataset)``` **
3. Training and testing the SerieseSleepNet
   + Change directory to ```./SeriesSleepNet```
   + Run main.py file
   
## Environment:
+ Python 3.8
+ Pytorch v1.9.0
+ numpy v1.22.2
+ scikit-learn v 0.24.2
+ tqdm v4.62.1

## Contact:
+ Minji Lee (Department of Brain and Cognitive Engineering, Korea University; minjilee@korea.ac.kr)
+ Heon-Gyu Kwak (Department of Artificial Intelligence, Korea University; hg_kwak@korea.ac.kr)

## License:
This project is released under the MIT License.
