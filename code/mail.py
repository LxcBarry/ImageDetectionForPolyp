# this python file is used to send a email to lxc
# when the training is done

import smtplib
from email.mime.text import MIMEText
from email.header import Header
# from datetime import datetime
import time

def send_to_me(net = 'resnet34+Unet',additinal=''):


    # 发送到你的邮箱

    mail_host = 'smtp.163.com'
    mail_user = 'byyx_1234@163.com'
    mail_pass = 'NGSZTBVMTAUEUEVD'
    sender = 'byyx_1234@163.com'
    receivers = ['980634323@qq.com','1979818737@qq.com'] #输入邮箱 '1979818737@qq.com'
    message = MIMEText(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+f"  {net} training done \n{additinal}" ,
                       'plain', 'utf-8')
    message['Subject'] = f'{net} 训练完成'
    message['From'] = sender
    message['To'] = receivers[0]
    try:
        smtpObj = smtplib.SMTP()
        
        #连接到服务器
        smtpObj.connect(mail_host,25)
        #登录到服务器
        smtpObj.login(mail_user,mail_pass)
        #发送
        smtpObj.sendmail(
            sender,receivers,message.as_string())
        #退出
        smtpObj.quit()
        print('success')
    except smtplib.SMTPException as e:
        print('error',e) #打印错误


if __name__ == '__main__':
    addtional = ""
    with open("log/resnet34/log.txt", 'r') as f:
        for line in f:
            addtional = addtional + line

    send_to_me(additinal= addtional)
    # send_to_me(additinal='test')