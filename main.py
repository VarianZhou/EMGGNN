from Data import EMGDataset
import TrainTheModel
import time

# define the links to all datasets
urls = ['https://springernature.figshare.com/ndownloader/files/25295225',
        'https://springernature.figshare.com/ndownloader/files/25295423',
        'https://springernature.figshare.com/ndownloader/files/25296089',
        'https://springernature.figshare.com/ndownloader/files/25296368',
        'https://springernature.figshare.com/ndownloader/files/25308197',
        'https://springernature.figshare.com/ndownloader/files/25296644',
        'https://springernature.figshare.com/ndownloader/files/25296788',
        'https://springernature.figshare.com/ndownloader/files/25309355',
        'https://springernature.figshare.com/ndownloader/files/25312253',
        'https://springernature.figshare.com/ndownloader/files/25313726',
        'https://springernature.figshare.com/ndownloader/files/25313324',
        'https://springernature.figshare.com/ndownloader/files/25312382',
        'https://springernature.figshare.com/ndownloader/files/25313078',
        'https://springernature.figshare.com/ndownloader/files/25313597',
        'https://springernature.figshare.com/ndownloader/files/25323104',
        'https://springernature.figshare.com/ndownloader/files/25323134',
        'https://springernature.figshare.com/ndownloader/files/25323215',
        'https://springernature.figshare.com/ndownloader/files/25323065',
        'https://springernature.figshare.com/ndownloader/files/25323257',
        'https://springernature.figshare.com/ndownloader/files/24350114']
names = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16',
         's17', 's18', 's19', 's20']

def execute():
    print('The project is executing')
    # Loading and Preparing the dataset
    raw = 'datasets/raw'
    # save = 'datasets/processed_distance_based'
    # save = 'datasets/combine_model_processed'
    save = 'datasets/processed'
    for i in range(20):
        try:
            start = time.time()
            dataset = EMGDataset(names[i], urls[i], raw, save)
            TrainTheModel.trainOnTheModel(dataset,names[i]+'_model')
            print(f'Execution ends, total running time is {time.time() - start}')
        except:
            continue

if __name__ == '__main__':
    execute()