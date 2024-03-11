from django.shortcuts import render, redirect
from login import models
from django.contrib import messages
from django.http import HttpResponse


def if_login(func):
    def alr_login(request, *args, **kwargs):
        studentID = request.session.get('studentID')
        # 判断是否已经登录了
        if studentID:
            return func(request, *args, **kwargs)
        else:
            return redirect('../login/')

    return alr_login


# Create your views here.
def login(request):
    if request.method == "POST":
        get_studentID = request.POST.get('studentID')
        get_password = request.POST.get('password')
        if_remember_password = request.POST.get('remember_password')
        try:
            password = models.UserInfo.objects.get(studentID=get_studentID).password
            Isfound = 1
        except:
            Isfound = 0

        if Isfound == 0:
            messages.error(request, "该用户不存在！")
            return render(request, 'login.html', {'studentID': get_studentID})
        else:
            if password == get_password:
                request.session['studentID'] = get_studentID
                request.session.set_expiry(1800)
                if if_remember_password == 'yes':
                    print("saved password")
                    request.session['password'] = get_password
                    request.session.set_expiry(1800)
                return redirect('../transfer/')
            else:
                messages.error(request, "密码错误！")
                return render(request, 'login.html', {'studentID': get_studentID})

    password = request.session.get('password')
    print(password)
    studentID = request.session.get('studentID')
    if studentID == 'None':
        studentID = ""
    if password == 'None':
        password = ""

    return render(request, 'login.html', {'studentID': studentID, 'password': password})


def register(request):
    # models.UserInfo.objects.create()
    if request.method == 'POST':
        get_username = request.POST.get('username')
        get_password = request.POST.get('password')
        get_repassword = request.POST.get('repassword')
        get_studentID = request.POST.get('studentID')
        get_group = request.POST.get('group')
        get_grade = request.POST.get('grade')

        if not (get_password and get_repassword and get_grade and get_group and get_username and get_studentID):
            messages.error(request, "请填写所有项目！")
            return render(request, 'register.html', locals())

        if get_password != get_repassword:
            messages.error(request, "两次输入的密码不一致！")
            return render(request, 'register.html', locals())

        try:
            result = models.UserInfo.objects.get(studentID=get_studentID)
            Isfound = 1
        except:
            Isfound = 0

        try:
            allowed_user = models.Users.objects.get(studentID=get_studentID)
            Isallow = 1
        except:
            Isallow = 0

        if (Isfound == 0) and (Isallow == 1):

            models.UserInfo.objects.create(username=get_username, password=get_password, studentID=get_studentID,
                                           group=get_group, grade=get_grade)
            messages.error(request, "注册成功！")
            return redirect('../login/')
        elif (Isfound == 0) and (Isallow == 0):
            messages.error(request, "该用户没有注册权限！")
        else:
            messages.error(request, "该用户已存在！")

        return render(request, 'register.html', locals())
    return render(request, 'register.html')


def register_return(request):
    return redirect('../../login/')


@if_login
def transfer(request):
    return render(request, 'transfer.html')


# def addUser(request):
#     models.Users.objects.create(studentID='example')
#     return HttpResponse('success')

def home(request):
    return redirect('../login/')


def page_not_found(request, exception):
    return render(request, 'error.html')
