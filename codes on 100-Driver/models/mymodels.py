import torch
import torch.nn as nn
import torchvision.models as model
import torch.nn.functional as F
class MSFIN_student11_without_ca(nn.Module):
    def __init__(self):
        super(MSFIN_student11_without_ca,self).__init__()
        

        '''for p in self.teacher.parameters():
            p.requires_grad = False'''
        
        temp_model = model.mobilenet_v2(pretrained=True)
        self.features = temp_model.features[:-8]
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=64, out_features=22, bias=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def getGAP(self,f):
        return self.avgpool(f).squeeze(-1).squeeze(-1)
    def getCAM(self,y,f,w,bias):
        camlists = []
        
        for i in range(y.size(0)):
            idx = y[i]
            tempw = w[idx]
            tempf = f[i]
            tempbias = bias[idx]
            c,d,e = tempf.size()
            
            tempf = tempw.unsqueeze(1).unsqueeze(1) * tempf
            tempf = tempf.sum(0) + tempbias
            cam = tempf.reshape(-1)

            c,d = tempf.size()
            cam = tempf.reshape(-1)
            cam = cam - cam.min().detach()
            cam = cam / cam.max().detach()
            #cam = (cam-cam.min())/(cam.max()-cam.min())
            cam = cam.reshape(d,e)
            camlists.append(cam)
        return torch.stack(camlists,0)
        
    def forward(self,x,y):
        
        f = self.features(x)
        avgf = self.getGAP(f)
        preds = self.fc(avgf)
        return preds,0

class MSFIN_student14_without_ca(nn.Module):
    def __init__(self):
        super(MSFIN_student14_without_ca,self).__init__()
        
        temp_model = model.mobilenet_v2(pretrained=True)
        self.features = temp_model.features[:-5]
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=96, out_features=22, bias=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def getGAP(self,f):
        return self.avgpool(f).squeeze(-1).squeeze(-1)
    def getCAM(self,y,f,w,bias):
        camlists = []
        
        for i in range(y.size(0)):
            idx = y[i]
            tempw = w[idx]
            tempf = f[i]
            tempbias = bias[idx]
            c,d,e = tempf.size()
            
            tempf = tempw.unsqueeze(1).unsqueeze(1) * tempf
            tempf = tempf.sum(0) + tempbias
            cam = tempf.reshape(-1)

            c,d = tempf.size()
            cam = tempf.reshape(-1)
            cam = cam - cam.min().detach()
            cam = cam / cam.max().detach()
            #cam = (cam-cam.min())/(cam.max()-cam.min())
            cam = cam.reshape(d,e)
            camlists.append(cam)
        return torch.stack(camlists,0)
        
    def forward(self,x,y):
        
        f = self.features(x)
        avgf = self.getGAP(f)
        preds = self.fc(avgf)
        
        return preds,0
    


class MSFIN_teacher(nn.Module):
    def __init__(self,trained_mobilenetv2):
        super(MSFIN_teacher,self).__init__()
        
        temp_model = trained_mobilenetv2
        self.features = temp_model.features
        self.fc = trained_mobilenetv2.fc
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def getGAP(self,f):
        return self.avgpool(f).squeeze(-1).squeeze(-1)
    def getCAM(self,y,f,w,bias):
        camlists = []
        
        for i in range(y.size(0)):
            idx = y[i]
            tempw = w[idx]
            tempf = f[i]
            tempbias = bias[idx]
            c,d,e = tempf.size()
            
            tempf = tempw.unsqueeze(1).unsqueeze(1) * tempf
            tempf = tempf.sum(0) + tempbias
            cam = tempf.reshape(-1)

            c,d = tempf.size()
            cam = tempf.reshape(-1)
            cam = (cam-cam.min())/(cam.max()-cam.min())
            cam = cam.reshape(d,e)
            camlists.append(cam)
        return torch.stack(camlists,0)
        
    def forward(self,x):
        f = self.features(x)
        avgf = self.getGAP(f)
        preds = self.fc(avgf)
        
        w = self.fc[1].weight
        b = self.fc[1].bias
        #print(f.shape,w.shape)
        cams = f.unsqueeze(1) * w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        cams = cams.sum(2) + b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return cams
class MSFIN_student14(nn.Module):
    def __init__(self,teacher):
        super(MSFIN_student14,self).__init__()
        
        self.teacher = teacher
        
        temp_model = model.mobilenet_v2(pretrained=True)
        self.features = temp_model.features[:-5]
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=96, out_features=22, bias=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def getGAP(self,f):
        return self.avgpool(f).squeeze(-1).squeeze(-1)
    def getCAM(self,y,f,w,bias):
        camlists = []
        
        for i in range(y.size(0)):
            idx = y[i]
            tempw = w[idx]
            tempf = f[i]
            tempbias = bias[idx]
            c,d,e = tempf.size()
            
            tempf = tempw.unsqueeze(1).unsqueeze(1) * tempf
            tempf = tempf.sum(0) + tempbias
            cam = tempf.reshape(-1)

            c,d = tempf.size()
            cam = tempf.reshape(-1)
            cam = cam - cam.min().detach()
            cam = cam / cam.max().detach()
            #cam = (cam-cam.min())/(cam.max()-cam.min())
            cam = cam.reshape(d,e)
            camlists.append(cam)
        return torch.stack(camlists,0)
        
    def forward(self,x,y):
        
        f = self.features(x)
        avgf = self.getGAP(f)
        preds = self.fc(avgf)
        
        self.teacher.eval()
        cams_t = self.teacher(x).detach()
        
        
        w = self.fc[1].weight
        b = self.fc[1].bias
        cams = f.unsqueeze(1) * w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        cams_s = cams.sum(2) + b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        a,b,c,d = cams_t.size()
        cams_t = cams_t.reshape(a*b,1,c,d)
        #print(cams_t.shape,cams_s.shape)
        #print(cams_t.shape)
        cams_t = F.interpolate(cams_t, size=(cams_s.size(2), cams_s.size(3)), mode='bilinear', align_corners=False)
        #print(cams_t.shape)
        cams_t = cams_t.squeeze(1)
        #print(cams_t.shape)
        cams_t = cams_t.reshape(a,b,cams_s.size(2), cams_s.size(3))
        
        #print(cams_s.shape,cams_t.shape)
        loss = nn.MSELoss()(cams_s,cams_t)
        return preds,loss

class MSFIN_student11(nn.Module):
    def __init__(self,teacher):
        super(MSFIN_student11,self).__init__()
        
        self.teacher = teacher

        '''for p in self.teacher.parameters():
            p.requires_grad = False'''
        
        temp_model = model.mobilenet_v2(pretrained=True)
        self.features = temp_model.features[:-8]
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=64, out_features=22, bias=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def getGAP(self,f):
        return self.avgpool(f).squeeze(-1).squeeze(-1)
    def getCAM(self,y,f,w,bias):
        camlists = []
        
        for i in range(y.size(0)):
            idx = y[i]
            tempw = w[idx]
            tempf = f[i]
            tempbias = bias[idx]
            c,d,e = tempf.size()
            
            tempf = tempw.unsqueeze(1).unsqueeze(1) * tempf
            tempf = tempf.sum(0) + tempbias
            cam = tempf.reshape(-1)

            c,d = tempf.size()
            cam = tempf.reshape(-1)
            cam = cam - cam.min().detach()
            cam = cam / cam.max().detach()
            #cam = (cam-cam.min())/(cam.max()-cam.min())
            cam = cam.reshape(d,e)
            camlists.append(cam)
        return torch.stack(camlists,0)
        
    def forward(self,x,y):
        
        f = self.features(x)
        avgf = self.getGAP(f)
        preds = self.fc(avgf)
        
        self.teacher.eval()
        cams_t = self.teacher(x).detach()
        
        
        w = self.fc[1].weight
        b = self.fc[1].bias
        cams = f.unsqueeze(1) * w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        cams_s = cams.sum(2) + b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        a,b,c,d = cams_t.size()
        cams_t = cams_t.reshape(a*b,1,c,d)
        #print(cams_t.shape,cams_s.shape)
        #print(cams_t.shape)
        cams_t = F.interpolate(cams_t, size=(cams_s.size(2), cams_s.size(3)), mode='bilinear', align_corners=False)
        #print(cams_t.shape)
        cams_t = cams_t.squeeze(1)
        #print(cams_t.shape)
        cams_t = cams_t.reshape(a,b,cams_s.size(2), cams_s.size(3))
        
        #print(cams_s.shape,cams_t.shape)
        loss = nn.MSELoss()(cams_s,cams_t)
        return preds,loss


