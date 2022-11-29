# -AI-MeoGeo-WeLM-
一个基于大语言模型(LLM)和文本转声音的模型（TTS）开发的玩意儿，代码比较拉跨。。。

如何使用：

1.首先打开conda prompt ，运行

pip install -r requirements.txt

2.下载模型，MeoGoe模型https://github.com/CjangCjengh/TTSModels

如果模型下架了可以参考https://space.bilibili.com/35285881/?spm_id_from=333.999.0.0，

或者私信我blibili账号获取模型https://space.bilibili.com/85747285/fans/follow?spm_id_from=333.1007.0.0
把下载好的模型放到models这个文件夹。


3.注册一个WeLM模型使用的授权码，详见微信WeLM模型https://welm.weixin.qq.com/docs/api/

3.参数设置，打开start.py文件，找到参数，具体内容参考微信WeLM模型https://welm.weixin.qq.com/docs/api/，

或者看参数后面的解释（里面包含了结合stable-diffusion的部分，但是现在效果还不行，不要去动它）

3.编辑 聊天记录.txt ，让AI生成符合你的品味的对话，可以尝试用galgame里面的对话，但不宜过多，上限100行

4.运行 start.py
