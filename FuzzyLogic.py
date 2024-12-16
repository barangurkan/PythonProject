def fuzzy_temperature(temperature):
    if temperature <= 20:
        cold = 1
        hot = 0
    elif 20 < temperature <= 30:
        cold = (30 - temperature) / 10
        hot = (temperature - 20) / 10
    else:
        cold = 0
        hot = 1

    return cold, hot



temperature = float(input("Odanın sıcaklığını gir (°C): "))
cold, hot = fuzzy_temperature(temperature)

print(f"Girilen sıcaklık: {temperature}°C")
print(f"Soğuk üyelik derecesi: {cold:.2f}")
print(f"Sıcak üyelik derecesi: {hot:.2f}")
