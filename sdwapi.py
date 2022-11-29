
import requests

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


