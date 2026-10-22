import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
def changeshape(x):#this x should be the x below. Whcih is:#by default x should be B,3,H,W(H,W is height and width while B represents how many pictures do we feed in) 
    B=x.shape[0]
    x_mid=x.view(B,3,224,14,16)
    x_mid2=torch.cat([x_mid[:,:,i,:,:] for i in range(224)],dim=3)
    x_mid3=x_mid2.view(B,3,196,256)
    x_final=torch.cat([x_mid3[:,i,:,:] for i in range(3)],dim=2)
    return x_final#result is B*196*768
#btw changespace could also be written by function"permute" and "unfold"
class pre(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Linear(768,768)
        self.cls=nn.Parameter(torch.zeros(1,1,768))#这里要写para而上面不用就是因为linear自动注册了W和B并且他们都可学习。
        self.position=nn.Parameter(torch.zeros(1,197,768))#不同的位置给的vec不同所以第二个位置是197而非1.不同的图一致所以第一个位置是1
    def forward(self,x):
        x_embedded=self.embedding(x)#nn.layer里面是写了call的，所以就这么调用
        #为什么cls不embed因为cls本身就是可学习参数。x在embed前根本不是可学习参数所以需要embed变成可学习的东西。
        cls=self.cls.expand(x.shape[0],1,768)
        #我自己一开始的歪门邪道是让self.cls加一个和x1一个形状的全0矩阵来强制广播
        input_0=torch.cat([cls,x_embedded],dim=1)
        input=input_0+self.position
        return input

class BLOCK(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(768)
        self.multiQ=nn.Parameter(torch.randn(768,768))
        self.multiK=nn.Parameter(torch.randn(768,768))
        self.multiV=nn.Parameter(torch.randn(768,768))
        self.comb12=nn.Linear(768,768)
        self.ln2=nn.LayerNorm(768)
        self.mlp=nn.Sequential(nn.Linear(768,3072),nn.GELU(),nn.Linear(3072,768))

#by default x should be B,3,H,W(H,W is height and width while B represents how many pictures do we feed in)
    def forward(self,x):
        B=x.shape[0]
        #第一轮残差

        input1=self.ln1(x)#B 197 768
        mulQ=(input1@self.multiQ).permute(0,2,1).view(B,12,64,197).permute(0,1,3,2)#注意这东西不用nn.para，因为nn.para和require grad是两码事。
        mulK=(input1@self.multiK).permute(0,2,1).view(B,12,64,197)#不调轴相当于直接转置
        mulV=(input1@self.multiV).permute(0,2,1).view(B,12,64,197).permute(0,1,3,2)
        mulA=torch.softmax((mulQ@mulK)/8,dim=3)
        mul=mulA@mulV
        Attentionfied=torch.cat([mul[:,i,:,:] for i in range(12)],dim=2)
        #下面是多头的最后一步让12也混合，虽然12本身的每一个就在结构上已经具有混合所有的能力了
        final_0=self.comb12(Attentionfied)#多头结束

        #残差接回
        final=final_0+x#同时作为新的残差

        #进mlp
        beforemlp=self.ln2(final)
        after=self.mlp(beforemlp)

        FINAL=after+final#这个FINAL将作为下一层的x（就是最开始那个x）
        return FINAL

class vit(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([pre()]+[BLOCK() for _ in range(12)])
        self.lnfinal=nn.LayerNorm(768)
        self.finalLinear=nn.Linear(768,10)
    def forward(self,input):
        for block in self.blocks:
            input=block(input)#这个时候完成了12次trans的迭代

        cls0=input[:,0,:]
        clsfn=self.lnfinal(cls0)
        logits=self.finalLinear(clsfn)
        return logits

#主函数：
model=vit()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
#下面是gpt帮我处理api，目前还不想在这种地方花太多时间

# CIFAR-10 图片 -> 你的 ViT 需要的输入
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 第一次运行自动下载；以后直接读取本地文件
train_set = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True
)

# 读出一个 batch
X, Y = next(iter(train_loader))

#gpt的工作到此结束
for epoch in range(2):
    for X,Y in train_loader:
        logits=model(changeshape(X))
        loss=F.cross_entropy(logits,Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())