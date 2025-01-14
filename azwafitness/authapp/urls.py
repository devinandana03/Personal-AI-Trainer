from django.urls import include,path
from . import views

urlpatterns = [
    path('',views.Home,name="Home"),
    path('signup',views.signup,name="signup"),
    path('login',views.handlelogin,name="handlelogin"),
    path('logout',views.handleLogout,name="handleLogout"),
    path('contact',views.contact,name="contact"),
    path('join',views.enroll,name="enroll"),
    path('profile',views.profile,name="profile"),
    path('gallery',views.gallery,name="gallery"),
    path('attendance',views.attendance,name="attendance"),  
    path('biceps/', views.biceps_pose, name='biceps_pose'),
    path('benchpress/', views.benchpress, name='benchpress'),
    path('chest/', views.chest, name='chest'),
    path('lateralraise/', views.lateralraise, name='lateralraise'),
    path('squats/', views.squats, name='squats'),
    path('legraise/', views.legraise, name='legraise'),
    path('shoulderpress/', views.shoulderpress, name='shoulderpress'),
    path('api/', include('exercise_log.urls')),
    path('russiantwist/', views.russiantwist, name='russiantwist'),
    path('tricepdips/', views.tricepdips, name='tricepdips'),
    path('triceppushdown/', views.triceppushdown, name='triceppushdown'),
    path('romaniandeadlift/', views.romaniandeadlift, name='romaniandeadlift'),
    path('lateralpulldown/', views.lateralpulldown, name='lateralpulldown'),
    path('CHEST/' , views.ExerciseLog, name='exercises'),
    path('exercisesbicep/', views.exercise_list, name='exercise_list'),
    path('curlbicep/', views.tobicepcurlpage, name='tobicepcurlpage'),
 # URL for biceps pose

    
]
