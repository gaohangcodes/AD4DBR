import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class MobilenetCA(nn.Module):
    def __init__(self):
        super(MobilenetCA,self).__init__()
        
        #model = MSFIN_basic()#models.mobilenet_v2(pretrained=True)
        #model.load_state_dict(torch.load(path))
        model = models.mobilenet_v2(pretrained=True)
        self.features = model.features
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=22, bias=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        '''self.map = nn.Sequential(
            nn.Conv2d(1280,1280,1,bias = False)
        )'''
    def getGAP(self,f):
        return self.avgpool(f).squeeze(-1).squeeze(-1)
    def getCAM(self,preds,w,f):
        #sa = getCAM(preds3,premodel.fc.weight,p17)
        camlists = []
        a,b = preds.size()
        for i in range(a):
            idx = preds[i].argmax(-1)
            #print(idx,w.shape,f.shape)
            tempw = w[idx]
            tempf = f[i]
            c,d,e = tempf.size()
            #weights = tempf.reshape(c,-1).mean(-1) * tempw
            #weights = F.softmax(weights)
            #print(tempw.shape,tempf.shape)
            tempf = tempw.unsqueeze(1).unsqueeze(1) * tempf
            tempf = tempf.sum(0)
            cam = tempf.reshape(-1)
            #print(cam.shape)
            #cam = F.softmax(cam,-1)
            #cam = cam / cam.sum(-1,True)
            c,d = tempf.size()
            cam = tempf.reshape(-1)
            cam = (cam-cam.min())/(cam.max()-cam.min())
            cam = cam.reshape(d,e)
            #print(cam.shape)
            camlists.append(cam)
        return torch.stack(camlists,0)
        
    def forward(self,x,labels):
        
        f = self.features(x)
        avgf = self.getGAP(f)
        preds = self.fc(avgf)
        
        #f = self.map(f)
        
        cam = f.unsqueeze(1) * self.fc[1].weight.unsqueeze(0).unsqueeze(3).unsqueeze(3)
        cams = cam.sum(2) + self.fc[1].bias.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        a,b,c,d = cams.size()
        cams = cams.reshape(a,b,-1)
        
        cams = cams - cams.min(-1)[0].unsqueeze(-1)
        
        cams = cams / (cams.max(-1)[0].unsqueeze(-1)+ 1e-30)
        
        
        cams0 = cams.reshape(a*b,c*d)
        
        N = 5
        topkv,topkp = cams0.topk(c*d,-1)
        loss1 = ((topkv[:,:N] - 1)**2).sum(-1)
        loss1 = ((topkv[:,N:] - 0)**2).sum(-1) + loss1
        loss1 = (loss1 / c / d).mean()

        temp = cams.unsqueeze(2) + cams.unsqueeze(1)
        eye = torch.triu(torch.ones(22,22)) - torch.eye(22)
        eye = eye.bool().cuda().unsqueeze(0).unsqueeze(3).expand(a,22,22,c*d)
        
        temp = temp[eye]
        loss3 = nn.ReLU()(temp-1).mean()#F.kl_div(temp.log(),torch.Tensor([[1]*49]*a).cuda()/49,reduction='batchmean')
        
        
        filtered_f = f.unsqueeze(1) * cams.reshape(a,22,c,d).unsqueeze(2)
        filtered_f = filtered_f.sum(-1).sum(-1) / c / d
        filtered_f = filtered_f.reshape(a,-1)
        co = F.cosine_similarity(filtered_f.unsqueeze(1),filtered_f.unsqueeze(0),-1)
        
        colabel = labels.unsqueeze(0) == labels.unsqueeze(1)
        colabel = colabel.float()
        #mask = colabel.sum(-1)>1
        #co = co[mask]
        #colabel = colabel[mask]
        #print(colabel[0])
        #print(co[0])
        #print(abcde)
        if colabel.size(0)>0:
            loss2 = F.kl_div(F.log_softmax(co,-1),colabel / colabel.sum(-1,True),reduction='batchmean')
        else:
            loss2 = 0

        return preds,loss1+loss2+loss3