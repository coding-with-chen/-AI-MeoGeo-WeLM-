# -*- coding: utf-8 -*-

# coding: utf-8
import make_sound as ms
import pandas as pd
import requests
import argparse
import os
from PIL import Image
# import matplotlib.pyplot as plt

#微信语言模型接口API
def WeLM(prompt = "给自己的猫咪取个特色的名字。",
        Authorization = 'cchi48mv9mc753cgtpf0',
         model = 'xl',
         max_tokens = 64,
         temperature = 0.85,
         top_p = 0.95,
         top_k = 50,
         n = 5,
         echo=False,
         stop = ',，.。'):
    

    # Up to 30 requests every 1 minute for each token.
    # Up to 1000000 characters can be generated every 24 hours.
    # The quota is reset every 24 hours (starting from the first request, within the next 24 hours).

    headers = {
        # 个人授权码
        'Authorization': Authorization,
    }
    
    json_data = {
        #提示词
        'prompt': prompt,
        'model': model,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'n': n,
        'echo': echo,
        'stop': stop,
    }
    
    response = requests.post('https://welm.weixin.qq.com/v1/completions', headers=headers, json=json_data)
    
    if n != 1:# 若返回多条消息，则返回最多字数的那一条
        stack = [[]]
        try:
            for i in eval(response.text)['choices']:
                content = (i['text'].split('\n'))[0]
                stack += [[content,len(content)]]
            stack = pd.DataFrame(stack)
        except:
            pass

        return stack.iloc[stack.iloc[:,1].argmax(),0]
    
    else:# 返回第一条
        return eval(response.text)['choices'][0]['text']
        
    
    
#  stable_diffusion_接口api
def stable_diffusion_api(prompt, negative_prompt, savepath,
                         post_url='https://a8ad5f586aec44f5.gradio.app/api/cyh', #发送post请求的网址
                         default_style1='None',default_style2='None',
                         sampling_steps=20,sampling_method="Euler a",cfg_scale=7,seed=-1,
                         hight=512,width=512):

    get_url = 'https://'+post_url.split('/')[2] + '/file=' ,#发送get请求的网址
    
    payload = { "data": [prompt,negative_prompt,default_style1,default_style2, sampling_steps, 
        sampling_method,False, False,1, 1, cfg_scale, seed, -1, 0, 0, 0, False, hight, width, 
        False, 0.7, 0, 0,"None",False, False, None,None,"Nothing", None, "Nothing",None,True,
        False,False, ]}#请求数据
    
    response = requests.post(url=post_url, json=payload) #发送post请求
    r = response.json() #解析网页
    # print('图片保存至',r['data'][0][0]['name'])
    a = requests.get(get_url[0]+r['data'][0][0]['name'])
    
    with open(savepath,'wb') as f:
        f.write(a.content)
        
def analyze_emotion(message):
    with open('情感.txt','r',encoding='utf-8') as f:
        a = f.read()
    a += (message+'\n'+'答案：')
    a = WeLM(prompt=a, Authorization=args.Authorization, model=args.Lan_model,
                max_tokens=args.max_tokens, temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k,n=args.n,stop=args.stop)
    return a

def draw_prompt(emotion_prompt,character):
    with open('场景.txt','r',encoding='utf-8') as f:
        a = f.read()
    prompt = character +','+ emotion_prompt +','+ a
    return prompt
    
    
def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    #语音模块参数
    parser.add_argument('--sound_models',  default='models/Cn_Ja_Nene + Nanami + Rong + Tang.pth', type=str, required=False, help='语音模型路径')
    parser.add_argument('--sound_config',  default='models/Cn_Ja_Nene + Nanami + Rong + Tang.json', type=str, required=False, help='语音模型配置文件路径')
    parser.add_argument('--out_path',      default='outputs/test.wav', type=str, required=False, help='音频wav格式输出路径')
    parser.add_argument('--language',      default='中文', type=str, required=False, help='语言，可选中文，英文，日文')
    parser.add_argument('--speaker_id',    default=1, type=int, required=False, help='语音模型人物序号')
    parser.add_argument('--speech_speed',  default=1.2, type=float, required=False, help='说话语速')
    parser.add_argument('--AIname',        default='妹妹', type=str, required=False, help='AI名字')
    parser.add_argument('--myname',        default='我', type=str, required=False, help='你的名字')
    
    #大语言模型参数
    parser.add_argument('--Authorization', default='cchi48mv9mc753cgtpf0', type=str, required=False, help='微信WeLM模型授权码，详见https://welm.weixin.qq.com/docs/api/')
    parser.add_argument('--Lan_model',     default='xl', type=str, required=False, help='微信WeLM模型，可选medium、 large、xl')
    parser.add_argument('--max_tokens',    default=32, type=int, required=False, help='WeLM模型token数')
    parser.add_argument('--temperature',   default=0.85, type=float, required=False, help='更高的temperature意味着模型具备更多的可能性。对于更有创造性的应用，可以尝试0.85以上，而对于有明确答案的应用')
    parser.add_argument('--top_p',         default=0.2, type=float, required=False, help='即从累计概率超过某一个阈值p的词汇中进行采样，所以0.1意味着只考虑由前10%累计概率组成的词汇。')
    parser.add_argument('--top_k',         default=50, type=int, required=False, help='范围，50~100，从概率分布中依据概率最大选择k个单词，建议不要过小导致模型能选择的词汇少。')
    parser.add_argument('--n',             default=3, type=int, required=False, help='默认值 1 返回的序列的个数')
    parser.add_argument('--stop',          default='\n', type=str, required=False, help=' 可选 默认值 null，停止符号')

    #stable-diffusion参数
    parser.add_argument('--prompt',         default='sister,loli', type=str, required=False, help='正面提示词')
    parser.add_argument('--negative_prompt',default='white, bad anatomy, liquid body, liquid tongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, lowres, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missing breasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fused ears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, fused animal ears, bad animal ears, poorly drawn animal ears, extra animal ears, liquid animal ears, heavy animal ears, missing animal ears, text, ui,', type=str, required=False, help='反面提示词')
    parser.add_argument('--img_savepath',   default='outputs/test.png', type=str, required=False, help='图片保存路径')
    parser.add_argument('--post_url',       default='https://4ffe3831cf3207a3.gradio.app/api/cyh', type=str, required=False, help='发送请求的网址')
    parser.add_argument('--sampling_steps', default= 20,  type=int, required=False, help='采样次数')
    parser.add_argument('--sampling_method',default='Euler a', type=str, required=False, help='采样方式')
    parser.add_argument('--cfg_scale',      default= 7, type=int, required=False, help='配置')
    parser.add_argument('--seed',           default= -1, type=int, required=False, help='随机种子')
    parser.add_argument('--hight',          default=512, type=int, required=False, help='图片高度')
    parser.add_argument('--width',          default=768, type=int, required=False, help='图片宽度')
    
    
    
    return parser.parse_args()





rp = os.path.dirname(os.path.realpath(__file__))#root path
args = set_args()
print(args.AIname+'正在苏醒......')

# img = Image.open(os.path.join(rp,'outputs/封面.png'))
# plt.figure("Image") # 图像窗口名称
# plt.imshow(img)
# plt.axis('off') # 关掉坐标轴为 off
# plt.title('AI-Gal') # 图像题目
# plt.show()
with open('聊天记录.txt',encoding='utf-8') as f:#展示之前的聊天记录
    content = f.read().split('\n')
    for i in content[:-1]:
        print(i)
    print('-----------------------以上是历史聊天记录-------------------------')
times = 3
while True:
    f = open('聊天记录.txt','r',encoding='utf-8')
    a = f.read()
    f.close()
    # 获取回复
    message = WeLM(prompt=a, Authorization=args.Authorization, model=args.Lan_model,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    top_p=args.top_p, top_k=args.top_k,n=args.n,stop=args.stop)
    
    if times == False:
        #分析情感
        emotion_prompt = analyze_emotion(message)
        #生成绘图提示词
        sdw_prompt = draw_prompt(emotion_prompt,args.prompt)
        # 生成图片
        stable_diffusion_api(prompt= args.prompt, negative_prompt=args.negative_prompt, savepath=os.path.join(rp,args.img_savepath),
                          post_url=args.post_url, default_style1='None',default_style2='None',
                          sampling_steps=args.sampling_steps,sampling_method=args.sampling_method
                          ,cfg_scale=args.cfg_scale, seed=args.seed,hight=args.hight,width=args.width)  
        
    
        #展示图片
        # img = Image.open(os.path.join(rp,args.img_savepath))
        # plt.figure("Image") # 图像窗口名称
        # plt.imshow(img)
        # plt.axis('off') # 关掉坐标轴为 off
        # plt.title('image') # 图像题目
        # plt.show()
        # times = 0

    
    #发声
    ms.main(model=os.path.join(rp,args.sound_models), config=os.path.join(rp,args.sound_config),language=args.language,
            text=message, speaker_id=args.speaker_id, 
            out_path=os.path.join(rp,args.out_path), name=args.AIname,speech_speed=args.speech_speed)


    #写入txt
    f = open('聊天记录.txt','a',encoding='utf-8')
    f.write(message+'\n'+args.myname+'：')
    f.close()
    
    times += 1
    
    #用户输入部分
    user_input = input(args.myname+'：')
    #若想离开了，就按q退出
    if user_input in ['q','Q','quit']:
        break
    f = open('聊天记录.txt','a',encoding='utf-8')
    f.write(user_input+'\n'+args.AIname+'：')
    f.close()





