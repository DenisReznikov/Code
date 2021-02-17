import sys 
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pandas as pd
import mainWindow 

from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication, QMainWindow
from plotly.graph_objects import Figure, Scatter
import plotly
import plotly.graph_objects as go


import numpy as np
from xgboost import XGBRegressor

class ExampleApp(QtWidgets.QMainWindow, mainWindow.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.__product = ""
        self.__xgb = XGBRegressor()
        self.setupUi(self)
        self.systems_comboBox.setEnabled(False)
        self.product_comboBox.setEnabled(False)
        self.groupBox_2.setEnabled(False)
        self.pushButton_star_modeling.setEnabled(False)
        self.tabWidget.setTabEnabled(1,False)
        self.pushButton_star_modeling.clicked.connect(self.star_modeling)
        self.load_pushButton.clicked.connect(self.browse_folder)
        self.systems_comboBox.activated.connect(self.load_product)
        self.product_comboBox.activated.connect(self.choose_product)

        self.pushButton_2.clicked.connect(self.predict_plot)





    def predict_plot(self):
        data = [[int(self.priority_spinBox.value()),  int(self.department_spinBox.value()),int(self.product_spinBox.value()),int(self.systems_spinBox.value())]] 
        df_tmp = pd.DataFrame(data, columns = ['priority', 'department', 'product', 'systems'])
        data = [[int(self.priority_spinBox_2.value()),  int(self.department_spinBox_2.value()),int(self.product_spinBox_2.value()),int(self.systems_spinBox_2.value())]] 
        df_tmp_2 = pd.DataFrame(data, columns = ['priority', 'department', 'product', 'systems']) 
 
        tmp = self.df_stock.loc[self.df_stock['ZK2'] == self.__product]
        
        plot = (tmp.resample('D')['Betroffenes System'].count().values)
        plot = plot[plot != 0]
        plot_1 = np.append(plot, self.__xgb.predict(df_tmp))
        plot_2 = np.append(plot, self.__xgb.predict(df_tmp_2))
        

        x = np.arange(1000)
        y = x**2
        fig = Figure(go.Scatter(y=plot_1,name='1'))
        fig.add_trace(go.Scatter( y=plot_2,name='2'))
        
        fig.add_vrect(x0=(len(plot)-1), x1=(len(plot)), 
              annotation_text="predict", annotation_position="top left",
              fillcolor="green", opacity=0.25, line_width=0)

        # we create html code of the figure
        html = '<html><body>'
        html += plotly.offline.plot(fig, output_type='div', include_plotlyjs='cdn')
        html += '</body></html>'

        # we create an instance of QWebEngineView and set the html code
        
        self.widget.setHtml(html)
        

    def star_modeling(self):
        self.list_product = []
        self.list_systems = []
        self.list_department = []
        self.list_priority = []
        
       
        time_range = ['M','W','D']
        self.df_train = pd.DataFrame()
        self.df_stock['Priorität'] = pd.to_numeric(self.df_stock['Priorität'], downcast='integer')   
        self.df_train['priority']  = self.df_stock.resample('D')['Priorität'].mean()
        self.df_train['department'] = self.df_stock.resample('D')['ORGEINHEIT'].nunique()
        
        self.df_train['product'] = self.df_stock.resample('D')['ZK2'].unique()
        self.df_train['y'] = self.df_stock.resample('D')['Betroffenes System'].count()
        
        zzkk = []
        zzkk_ = []
        dep_list = []
        
        for index, row in self.df_train.iterrows():
            zzkk_.extend(set(zzkk))
            l = ((row['product']))
            for j in l:
                zzkk_.append(j)

            if set(zzkk) == set(zzkk_):
                dep_list.append(0)
            else:
                
                dep_list.append(len(set(zzkk_)) - len(set(zzkk)))
                zzkk.extend(set(zzkk_))
        self.df_train['product'] = dep_list
        
        self.df_train['systems'] = self.df_stock.resample('D')['Betroffenes System'].unique()
        zzkk = []
        zzkk_ = []
        dep_list = []
        for index, row in self.df_train.iterrows():
            zzkk_.extend(set(zzkk))
            l = ((row['systems']))
            for j in l:
                zzkk_.append(j)

            if set(zzkk) == set(zzkk_):
                dep_list.append(0)
            else:
                
                dep_list.append(len(set(zzkk_)) - len(set(zzkk)) )
                zzkk.extend(set(zzkk_))
        self.df_train['systems'] = dep_list
        self.df_train.dropna(inplace=True)

        y_train = self.df_train['y']
        X_train = self.df_train.drop(['y'], axis=1)
        self.__xgb.fit(X_train, y_train)

        self.tabWidget.setTabEnabled(1,True)
    
        



    def check_department(self, state):
        if state == Qt.Checked:
            self.department_spinBox_2.setEnabled(True)
        else:
            self.department_spinBox_2.setEnabled(False)


    def check_new_product(self, state):
        if state == Qt.Checked:
             self.product_spinBox_2.setEnabled(True)
        else:
            self.product_spinBox_2.setEnabled(False)


    def check_new_system(self, state):
        if state == Qt.Checked:
            self.systems_spinBox_2.setEnabled(True)
        else:
            self.systems_spinBox_2.setEnabled(False)

    
    def check_priority(self, state):
        if state == Qt.Checked:
            self.priority_spinBox_2.setEnabled(True)
        else:
            self.priority_spinBox_2.setEnabled(False)


    def load_product(self):
        df_tmp = self.df_stock.loc[self.df_stock["Betroffenes System"]==self.systems_comboBox.currentText()]
        self.product = list(df_tmp['ZK2'].unique())
        
        self.systems = self.systems_comboBox.currentText()
        self.product_comboBox.clear()
        for i in self.product:
            
            if isinstance(i, float):
                continue
            else:
                self.product_comboBox.addItem(i)
        self.product_comboBox.setEnabled(True)


    def choose_product(self):
        self.__product = self.product_comboBox.currentText()
        self.groupBox_2.setEnabled(True)
        self.pushButton_star_modeling.setEnabled(True)

        

    def browse_folder(self):
        if self.excel_radioButton.isChecked():
            file, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', './', "Excel (*.xlsx)")
            if file == '':
                return
                #TODO: 
            self.df_stock = pd.read_excel(file)
        else:
            file, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', './', "Csv ( *.csv )")
            if file == '':
                return
                #TODO: 
            self.df_stock = pd.read_csv(file,sep=";",encoding='cp1252')

        self.df_stock.drop(self.df_stock.tail(1).index, inplace=True)
        self.df_stock = self.df_stock.loc[self.df_stock['Service-Beginn'] != "Service-Beginn" ]
        self.df_stock['Service-Beginn'] = pd.to_datetime(self.df_stock['Service-Beginn'].copy())
        self.df_stock = self.df_stock.loc[self.df_stock['Service-Beginn'].dt.year==2020]
       
        self.df_stock.sort_values(by=['Service-Beginn'],inplace=True)
        self.df_stock.set_index('Service-Beginn',inplace=True)
        self.systems = list(self.df_stock['Betroffenes System'].unique())
        self.df_stock.drop(self.df_stock.tail(1).index, inplace=True)
        for i in self.systems:
            self.systems_comboBox.addItem(i)

        
        self.systems_comboBox.setEnabled(True)
        


def main():
    app = QtWidgets.QApplication(sys.argv) 
    window = ExampleApp() 
    window.show() 
    app.exec_() 

if __name__ == '__main__':
    main()


