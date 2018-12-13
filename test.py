import datetime as d
time_start = d.datetime.now()
print(time_start)
print(time_start.strftime("%d.%m.%Y-%H:%M:%S"))

time_end = d.datetime.now()
print(time_start)
dauer = (time_end - time_start)
print(dauer)
list = [1,2,"3",55]
with open('Result_Daten\\test\\test.txt', 'x'and"a") as f:
    for i in list:
     print(i, file=f)