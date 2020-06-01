# coding=utf-8
import wx
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
import recognition_pic as rp #图片检测
import recognition_camera as rc  #相机检测
import webbrowser
from pyecharts import Pie

class Frame(wx.Frame):
    # 界面框架初始化
    def __init__(self):
        wx.Frame.__init__(self,parent=None,title="人脸表情识别系统", size=(900, 1200))
        self.locale = wx.Locale(wx.LANGUAGE_ENGLISH)
        # 使用GridBagSizer布局管理部件进行框架布局
        self.grid_sizer=wx.GridBagSizer(0,0)  

        self.filename='../data/default.png'
        # 显示标题面板
        self.panel_top=wx.Panel(self)
        self.panel_top.SetMinSize((700,200))
        self.grid_sizer.Add(self.panel_top,pos=(0,0),span=(1,2),flag=wx.EXPAND)
        self.img_top=wx.Image('../data/title.png',wx.BITMAP_TYPE_ANY).Scale(900, 200)
        wx.StaticBitmap(self.panel_top,pos=(0,0),bitmap=wx.Bitmap(self.img_top))
        # 显示图片展示面板
        self.panel_img=wx.Panel(self)
        self.panel_img.SetMinSize((700,350))
        self.panel_img.SetBackgroundColour((255, 205, 211))
        self.grid_sizer.Add(self.panel_img,pos=(1,0),span=(1,1),flag=wx.EXPAND)
        self.setImage(self.filename,'0')
#        self.img_default=wx.Image('../temp/default.png',wx.BITMAP_TYPE_ANY)
#        wx.StaticBitmap(self.panel_left,pos=(100,50),bitmap=wx.Bitmap(self.img_default))
        # 显示按钮面板
        self.panel_btn=wx.Panel(self)
        self.panel_btn.SetMinSize((700,150))
        self.panel_btn.SetBackgroundColour((255, 205, 211))
        self.grid_sizer.Add(self.panel_btn,pos=(2,0),span=(1,1),flag=wx.EXPAND)
        # 位图按钮
        self.up=wx.Image('../data/up.png',wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.upBtn=wx.BitmapButton(self.panel_btn,-1,self.up,pos=(150,50))
        self.upBtn.Bind(wx.EVT_BUTTON,self.upBtnEvent)
        self.start=wx.Image('../data/start.png',wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.startBtn=wx.BitmapButton(self.panel_btn,-1,self.start,pos=(400,50))
        self.startBtn.Bind(wx.EVT_BUTTON,self.startBtnEvent)
        # 右边按钮
        self.panel_right=wx.Panel(self)
        self.panel_right.SetMinSize((200,500))
        self.panel_right.SetBackgroundColour((255, 205, 211))
        self.grid_sizer.Add(self.panel_right,pos=(1,1),span=(2,1),flag=wx.EXPAND)

        self.cama = wx.Image('../data/ca.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.camaBtn = wx.BitmapButton(self.panel_right, -1, self.cama, pos=(0, 50))
        self.camaBtn.Bind(wx.EVT_BUTTON, self.startBtnEvent_cam)

        self.chart_show = wx.Image('../data/show.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.chart_showBtn = wx.BitmapButton(self.panel_right, -1, self.chart_show, pos=(0, 200))
        self.chart_showBtn.Bind(wx.EVT_BUTTON, self.chart_show1)


        self.SetSizer(self.grid_sizer)
        self.Fit()

    # 显示图片，flag表示是否为默认图片
    def setImage(self,path,flag):
        self.img_default=wx.Image(path,wx.BITMAP_TYPE_ANY).Rescale(500,300).ConvertToBitmap()

#        source = cv2.imread(path, cv2.IMREAD_COLOR)
#        img = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
#        h, w = img.shape[:2]
#        wxbmp = wx.BitmapFromBuffer(w, h, img)
#        size = wxbmp.GetWidth(),wxbmp.GetHeight()
        if flag=='0':
            self.img=wx.StaticBitmap(self.panel_img,pos=(100,50),bitmap=self.img_default)
        else:
            self.img.SetBitmap(wx.Bitmap(self.img_default))

    # 上传图片按钮事件，点击后弹出文件对话框，获取所选图片的路径
    def upBtnEvent(self,event):
        dlg=wx.FileDialog(self, message="Choose a file", defaultDir='../data',defaultFile='',style=wx.FD_OPEN, wildcard="*.*",pos=wx.DefaultPosition)
        if dlg.ShowModal()==wx.ID_OK:
            self.filename=dlg.GetPath()
            self.setImage(self.filename,'1')
            print(self.filename)

    # 开始检测按钮事件，点击后调用load_meta_cnn进行识别检测并展示结果
    def startBtnEvent(self, event):
        if rp.main(self.filename)==1:
            # wx.ProgressDialog(title='检测中', message='waiting...', maximum=100, parent=self,style=wx.PD_CAN_ABORT)
            f=open('../data/cv/1.txt')
            line=f.readline()
            d=eval(line)
            self.y = [d['生气'], d['尴尬'], d['惊讶'], d['开心'], d['伤心'], d['惊讶'], d['自然']]
            self.setPlot(self.y)
            self.drawPie(self.y)
            self.setImage('../data/cv/pic/test.jpg','1')
        else:
            dlg = wx.MessageDialog(None, "对不起没有识别到人脸",
                                   '警告提示',
                                   wx.OK | wx.ICON_QUESTION)
            retCode = dlg.ShowModal()

            dlg.Destroy()

        # 开始检测按钮事件，点击后调用load_meta_cnn使用camara进行识别检测并展示结果
    def startBtnEvent_cam(self, event):
        if rc.main(self.filename)==1:
            # wx.ProgressDialog(title='检测中', message='waiting...', maximum=100, parent=self,style=wx.PD_CAN_ABORT)
            f = open('../data/cv/1.txt')
            line = f.readline()
            d = eval(line)
            self.y = [d['生气'], d['尴尬'], d['惊讶'], d['开心'], d['伤心'], d['惊讶'], d['自然']]
            self.setPlot(self.y)
            self.drawPie(self.y)
            self.setImage('../data/cv/pic/test.jpg', '1')
        else:
            print(21)
            dlg = wx.MessageDialog(None, "对不起没有识别到人脸",
                                   '警告提示',
                                   wx.OK | wx.ICON_QUESTION)
            retCode = dlg.ShowModal()
            dlg.Destroy()
    def chart_show1(self, event):
        webbrowser.open("render.html")

    def drawPie(self,y=[1,1,1,1,1,1,1]):

        position = ['生气', '尴尬', '害怕', '开心', '伤心', '惊讶', '自然']
        pie = Pie("表情分析", title_pos='center', width=900, title_text_size=20)

        pie.add("指数", position, y,

                center=[50, 50],

                is_random=1,

                radius=[30, 75],

                rosetype='radius',

                is_legend_show=0,

                is_label_show=1, label_text_size=20)

        pie.render()
    # 显示条形图
    def setPlot(self,y=[1,1,1,1,1,1,1]):
        self.fig,self.axe=plt.subplots()
        #self.canvas=FigureCanvasWxAgg(self.panel_right,-1,self.fig)
        self.x=np.arange(7)
        self.y=y
        plt.bar(self.x,self.y,facecolor='#ffd07e')
        for x_, y_ in zip(self.x, self.y):
            plt.text(x_, y_, '{:.2f}'.format(y_), ha='center', va='bottom',color='#ff8776',fontsize=11)
        plt.xticks([0,1,2,3,4,5,6],[r'$angry$',r'$disgust$',r'$fear$',r'$happy$',r'$sad$',r'$surprise$',r'$neutral$'])


app = wx.App()
frame = Frame()
frame.Show()
app.MainLoop()
