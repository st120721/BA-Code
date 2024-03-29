"""
    feature_extracion_wavelet:

    Das Modul ist aufgebaut, um die Features aus Wellenform-Daten zu extrahieren.
    Toolbox: pywt für Wavelet-Transformation

    @version: 1.3

"""

import numpy as  np
import pandas as pd
from scipy import stats
import pywt
import os


class Feature_Extraction_Wavelet:
    """
          Feature_Extraction_Wavelet:

          Hauptklasee vom Modul "feature_extracion_wavelet"
          Und eine Unterklasse "Calculate_Feature" ist aufgebaut,um die Features zu rechnen.

          Attributes:
              list_extacion_methode: eine Liste von Wavelettype
              list_features: eine Liste von Features
              folder_path: das Pfad vom Ordner ,in dem die Ergebnisse gespeichert werden.
              raw_data: Rohdaten
              features: features(Datentype:DataFrame)
              labels: labels(Datentype:DataFrame)
              features_and_labels: features and labels(Datentype:DataFrame)

      """

    list_extacion_methode =["Discrete Wavelet Transformation(DWT)","Wavelet Packet(WP)"]
    list_features =["energy","shanon_entropy","mean","standard deviation","RMS","kurtosis","skewness"]
    # list_features=["Log-energy entropy","Interquartile range","Form factor","Crest-factor"]

    def __init__(self,folder_path,data_name,transformation_name,transformation_level):
        """
            Konstruktionsfunktion von der Klasse

            :param folder_path: das Pfad vom Ordner ,in dem die Ergebnisse gespeichert werden.
            :param data_name: Name der Dokumente von Rohndaten
            :param transformation_name: Wavelettype(zur zeit "WP"order"DWT")
            :param transformation_level: Schicten von Transformation

        """
        # Ein Ordner wird hergestellt,when der Ordner existiert nicht.
        isExists = os.path.exists(folder_path)
        if not isExists:
            os.makedirs(folder_path)
        self.folder_path =folder_path
        self.transformation_name = transformation_name
        self.transformation_level = transformation_level
        self.data_name = data_name
        self.raw_data=self.load_data()
        self.features,self.labels,self.features_and_labels = self.features_extraction()
        self.wirte_to_csv()

    def load_data(self):
        """
            lade Daten

            :return: Rohdaten

        """

        path ="Raw_Daten\\"+self.data_name

        # Ein Ordner wird hergestellt,when der Ordner existiert nicht.
        isExists = os.path.exists(path)
        if not isExists:
            path = self.data_name

        raw_data =pd.read_csv(path)
        return raw_data

    def wirte_to_csv(self):
        """
             Features und Labels in Excel speichern

        """
        output_data = self.features_and_labels
        output_name ="FE_"+self.transformation_name+"("+self.data_name+")"+".csv"
        output_path=self.folder_path+"\\"+output_name
        output_data.to_csv(output_path,index=False)


    def features_extraction(self):
        """
            Methode für Feature Extraktion von aller Beispiele

            :return: features,labels,features_and_labels

        """
        features =[]
        data = self.raw_data
        labels = pd.DataFrame(data.ix[:, data.shape[1] - 1])
        labels.columns = ["label"]
        data =data.drop([str(data.shape[1]-1)],axis=1)
        for m in range(0,data.shape[0]):

            data_row= data[m:m+1].ix[:, 0:data.shape[1]]
            data_row = np.asarray(data_row).tolist()[0]

            if self.transformation_name =="WP":
                coeff =self.Wavelet_Packet.wavelet_packet_transformation(data_row,self.transformation_level)
            elif self.transformation_name =="DWT":
                coeff=self.Discrete_Wavelet_Transformation.wavelet_transformation(data_row,self.transformation_level)

            feature_single =np.array(self.Calculate_Feature.feature_extraction_singe_instance(coeff))
            features.append(feature_single)

        features =pd.DataFrame(features)
        features_and_labels = features.join(labels)
        return features,labels,features_and_labels

    class Calculate_Feature:
        """
            Calculate_Features:

            Die Klasse ist aufgebaut, um die Features von einem Beispiel zu rechnen.

        """
        @staticmethod
        def feature_energy(coeff):
            """
                Energie rechnen

                :param coeff: die Koeffizienten von einem Beispiel nach Wavelet-Transformation

                :return: energy

            """
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
            """
                features von einem Beispiel rechnen und in einer Liste speichern

                :param coeff: die Koeffizienten von einem Beispiel nach Wavelet-Transformation

                :return: features_single_instance, features von einem Beispiel

            """
            feature_energy =Feature_Extraction_Wavelet.Calculate_Feature.feature_energy(coeff)
            feature_shanon_ent =Feature_Extraction_Wavelet.Calculate_Feature.feature_shanon_entropy(coeff)
            feature_mean =Feature_Extraction_Wavelet.Calculate_Feature.feature_mean(coeff)
            feature_std =Feature_Extraction_Wavelet.Calculate_Feature.feature_standard_deviation(coeff)
            feature_kur =Feature_Extraction_Wavelet.Calculate_Feature.feature_kurtosis(coeff)
            feature_ske =Feature_Extraction_Wavelet.Calculate_Feature.feature_skewness(coeff)
            feature_rms=Feature_Extraction_Wavelet.Calculate_Feature.feature_RMS(coeff)
            feature_single_instance =feature_energy+feature_shanon_ent+feature_mean+feature_std+feature_rms+feature_kur+feature_ske
            return feature_single_instance

    class Wavelet_Packet:
        """
            Wavelet_Packet:

            Wavelet Packet Transformation

        """
        @staticmethod
        def wavelet_packet_transformation(data,level_num):
            """
                Wavelet Packet Transformation,

                :param data: die Daten von einem Beispiel
                :param level_num: Schichten von Wavelet-Transformation
                :return: die Koeffizienten von einem Beispiel nach Wavelet-Transformation

            """
            # wavelet – Wavelet used in DWT decomposition and reconstruction
            # mode_name – Signal extension mode for the `dwt` and `idwt` decomposition and reconstruction functions.
            wavelet = 'db4'
            mode ='symmetric'

            maxlevel = pywt.dwt_max_level(len(data),"db4")
            if maxlevel < level_num:
                print("level_num >maxlevel")

            wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode=mode)

            nodes = wp.get_level(level_num, order="natural")
            coeff = np.array([n.data for n in nodes], 'd')

            # values = abs(values)
            return coeff

    class Discrete_Wavelet_Transformation:
        """
            Discrete_Wavelet_Transformation:

            Discrete Wavelet Transformation(DWT)

        """
        @staticmethod
        def wavelet_transformation(data,level_num):
            """
                Wavelet Packet Transformation,

                :param data: die Daten von einem Beispiel
                :param level_num: Schichten von Wavelet-Transformation
                :return: die Koeffizienten von einem Beispiel nach Wavelet-Transformation

            """
            # wavelet – Wavelet to use
            # mode – Signal extension mode, see `Modes` (default: 'symmetric')
            wavelet="db4"
            mode = "symmetric"
            coeff_temp =pywt.wavedec(data=data, wavelet=wavelet, mode=mode, level=level_num)

            coeff = pd.DataFrame(coeff_temp[0]).T

            for num in range(0, level_num):
                     coeff = pd.concat((coeff,pd.DataFrame(coeff_temp[num+1]).T), ignore_index=True)


            coeff = np.array(coeff)
            return coeff
