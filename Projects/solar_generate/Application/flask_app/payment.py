def calc_payment(month, power_use):
    if month in [7, 8]:
        if power_use <= 300:
            payment = 910 + power_use*100.6
        elif power_use <= 450:
            payment = 1600 + 300*100.6 + (power_use-300)*195.2
        elif power_use <= 1000:
            payment = 7300 + 300*100.6 + 150*195.2 + (power_use-450)*287.9
        else:
            payment = 7300 + 300*100.6 + 150*195.2 + 550*287.9 + (power_use-1000)*716.8

    else:
        if power_use <= 200:
            payment = 910 + power_use*100.6
        elif power_use <= 400:
            payment = 1600 + 200*100.6 + (power_use-200)*195.2
        else:
            if month in [1, 2, 12]:
                if power_use <= 1000:
                    payment = 7300 + 200*100.6 + 200*195.2 + (power_use-400)*287.9
                else:
                    payment = 7300 + 200*100.6 + 200*195.2 + 600*287.9 + (power_use-1000)*716.8
            else:
                payment = 7300 + 200*100.6 + 200*195.2 + (power_use-400)*287.9

    total_payment = ((payment + power_use*7.30) * 1.137) // 10 * 10

    return total_payment