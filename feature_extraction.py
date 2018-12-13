import numpy as  np
import pandas as pd
from scipy import stats
import pywt
import os

class Feature_Extraction:
    list_extacion_methode =["Discrete Wavelet Transformation(DWT)","Wavelet Packet(WP)"]
    list_features =["energy","shanon_entropy","mean","standard deviation","kurtosis","skewness"]
    # list_features=["RMS","Log-energy entropy","Interquartile range","Form factor","Crest-factor"]
    def __init__(self,test_name,data_name,transformation_name,transformation_level):

        self.test_name =test_name
        self.transformation_name = transformation_name
        self.transformation_level = transformation_level
        self.test_data_name = data_name
        self.raw_data=Feature_Extraction.load_data(data_name)
        self.features,self.labels,self.features_and_labels = self.features_extraction()
        self.output_path=self.wirte_to_csv(self.features_and_labels)

    @staticmethod
    def load_data(data_name):
        path ="Raw_Daten\\"+data_name
        raw_data =pd.read_csv(path)
        return raw_data

    def wirte_to_csv(self,output_data):
        # features_and_labels =self.feature_and_label(data)
        # features_and_labels.to_csv('FeaturesAndLabelsWP1700.csv',index=False)

        isExists = os.path.exists("Result_Daten")
        if not isExists:
            os.makedirs("Result_Daten")
        output_path = "Result_Daten\\" +self.test_name
        isExists = os.path.exists(output_path)
        if not isExists:
            os.makedirs(output_path)

        output_name ="FE_"+self.transformation_name+"("+self.test_data_name+")"+".csv"
        output_path=output_path+"\\"+output_name
        output_data.to_csv(output_path,index=False)
        return output_path

    def features_extraction(self):
        feature_name_list =[]
        features =[]
        data = self.raw_data
        labels = pd.DataFrame(data.ix[:, data.shape[1] - 1])
        labels.columns = ["label"]
        data =data.drop([str(data.shape[1]-1)],axis=1)
        for m in range(0,data.shape[0]):
            # label =data.ix[m,data.shape[1]-1].tolist()
            # print(label.__class__)
            data_row= data[m:m+1].ix[:, 0:data.shape[1]]
            data_row = np.asarray(data_row).tolist()[0]
            if self.transformation_name =="WP":
                coeff =Wavelet_Packet.wavelet_packet_transformation(data_row,self.transformation_level)
            elif self.transformation_name =="DWT":
                coeff=Discrete_Wavelet_Transformation.wavelet_transformation(data_row,self.transformation_level)

            feature_single =np.array(Calculate_Feature.feature_extraction_singe_instance(coeff))
            features.append(feature_single)

        features =pd.DataFrame(features)
        features_and_labels = features.join(labels)
        return features,labels,features_and_labels

class Calculate_Feature:


    # data: single instance,row in matrix
    # data:class 'numpy.ndarray'
    @staticmethod
    def feature_energy(coeff):
        energy_array =[]
        for m in range(0,coeff.shape[0]):

            coeff_list =coeff[m,:]
            coeff_list = coeff_list[~np.isnan(coeff_list)]

            squ = np.square(coeff_list)
            sum =np.sum(squ)
            energy_array.append(sum)
        return energy_array

    @staticmethod
    def feature_shanon_entropy(coeff):
        ent_array = []
        for m in range(0, coeff.shape[0]):

            coeff_list = coeff[m, :]
            coeff_list = coeff_list[~np.isnan(coeff_list)]

            squ = np.square(coeff_list)
            ent_temp = stats.entropy(squ)
            ent_array.append(ent_temp)
        return ent_array

    @staticmethod
    def feature_mean(coeff):
        mean_array=[]
        for m in range(0, coeff.shape[0]):

            coeff_list = coeff[m, :]
            coeff_list = coeff_list[~np.isnan(coeff_list)]

            mean_temp=np.mean(coeff_list)
            mean_array.append(mean_temp)
        return mean_array

    @staticmethod
    def feature_standard_deviation(coeff):
        std_array = []
        for m in range(0, coeff.shape[0]):

            coeff_list = coeff[m, :]
            coeff_list = coeff_list[~np.isnan(coeff_list)]

            std_temp = np.std(coeff_list)
            std_array.append(std_temp)
        return std_array

    @staticmethod
    def feature_RMS(coeff):
        RMS_array = []
        for m in range(0, coeff.shape[0]):
            coeff_list = coeff[m, :]
            coeff_list = coeff_list[~np.isnan(coeff_list)]


            RMS_temp = ((np.square(coeff_list)).mean())**0.5
            RMS_array.append(RMS_temp)
        return RMS_array


    @staticmethod
    def feature_kurtosis(coeff):
        kur_array = []
        for m in range(0, coeff.shape[0]):

            coeff_list = coeff[m, :]
            coeff_list = coeff_list[~np.isnan(coeff_list)]
            kur_temp = stats.kurtosis(coeff_list)
            kur_array.append(kur_temp)
        return kur_array

    @staticmethod
    def feature_skewness(coeff):
        skew_array = []
        for m in range(0, coeff.shape[0]):

            coeff_list = coeff[m, :]
            coeff_list = coeff_list[~np.isnan(coeff_list)]

            skew_temp = stats.skew(coeff_list)
            skew_array.append(skew_temp)
        return skew_array

    @staticmethod
    def feature_extraction_singe_instance(coeff):
        feature_energy =Calculate_Feature.feature_energy(coeff)
        feature_shanon_ent =Calculate_Feature.feature_shanon_entropy(coeff)
        feature_mean =Calculate_Feature.feature_mean(coeff)
        feature_std =Calculate_Feature.feature_standard_deviation(coeff)
        feature_kur =Calculate_Feature.feature_kurtosis(coeff)
        feature_ske =Calculate_Feature.feature_skewness(coeff)
        feature_rms=Calculate_Feature.feature_RMS(coeff)
        feature_single_instance =feature_energy+feature_shanon_ent+feature_mean+feature_std+feature_rms+feature_kur+feature_ske
        return feature_single_instance

class Wavelet_Packet:
    # data: single instance,row in raw data
    # return:class 'numpy.ndarray'
    @staticmethod
    def wavelet_packet_transformation(data,level_num):
        wavelet = 'db2'
        mode_name ='symmetric'
        # Construct wavelet packet

        maxlevel = pywt.dwt_max_level(len(data),"db4")

        wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode=mode_name)

        nodes = wp.get_level(level_num, order="natural")
        coeff = np.array([n.data for n in nodes], 'd')

        # values = abs(values)
        return coeff

class Discrete_Wavelet_Transformation:
    @staticmethod
    def wavelet_transformation(data,level_num):
        wavelet="db4"
        mode_name = "symmetric"
        coeff_temp =pywt.wavedec(data=data, wavelet=wavelet, mode=mode_name, level=level_num)

        coeff = pd.DataFrame(coeff_temp[0]).T


        for num in range(0, level_num):
                 coeff = pd.concat((coeff,pd.DataFrame(coeff_temp[num+1]).T), ignore_index=True)


        coeff = np.array(coeff)
        return coeff

#
# test = Feature_Extraction(test_name="test",data_name="TestData_1700.csv",
#                           transformation_name="WP",transformation_level =4)
# print(test.features)
# print(test.features_and_labels)
