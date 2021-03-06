import chardet
import time


def brute_xor_key(string):
    for i in range(-10000, 100000):
        a = ''
        for j in string:
            masked = ord(j) ^ i
            if masked < 21 or masked > 126:
                a = ''
                break
            a += chr(masked)
        if a != '':
            print(a)
            time.sleep(1)
            print('=' * 70)


def find_key(string):
    out = chardet.detect(str.encode(string))
    print(out)
    return out


def main():
    string = 'JABLAGUAYgBtAHcAbwBzAGsAeQB2AG8AZAA9ACcATwBzAGoAZQBjAGkAaQBwACcAOwAkAFIAZABhAHIAZgB6AG4AeABhAGsAbQBiACAAPQAgACcAOAA5ADQAJwA7ACQASABvAHAAcAB4AHAAagB4AGUAeQBoAHYAPQAnAFUAYQBnAGUAZQBjAGMAZQAnADsAJABMAGsAeQBqAG0AdwB2AHkAPQAkAGUAbgB2ADoAdQBzAGUAcgBwAHIAbwBmAGkAbABlACsAJwBcACcAKwAkAFIAZABhAHIAZgB6AG4AeABhAGsAbQBiACsAJwAuAGUAeABlACcAOwAkAEkAYQBoAHkAZABsAHYAZwBjAHMAPQAnAEMAdgBhAGYAZQBmAHEAZQAnADsAJABaAHkAcABwAHAAbQBvAHUAZABiAHEAbAA9ACYAKAAnAG4AJwArACcAZQAnACsAJwB3AC0AbwBiAGoAZQBjAHQAJwApACAAbgBlAFQALgB3AGUAYgBDAEwAaQBFAE4AdAA7ACQARgBiAGcAeAB6AG4AdABsAGEAbwBmAD0AJwBoAHQAdABwADoALwAvAHMAcgBpAHMAdQByAGUAbgBhAC4AYwBvAG0ALwB0AHIAYQBkAGUAbQBhAHIAawAvAGMANQA1ADMAYwAvACoAaAB0AHQAcAA6AC8ALwBuAGUAeABzAG8AbABnAGUAbgAuAGMAbwBtAC8AYwA4AHQAcwB6ADMAMAAvAHAAegBiAHkALwAqAGgAdAB0AHAAOgAvAC8AdgBlAGQAYQBuAHMAaABzAG8AZgB0AC4AYwBvAG0ALwBlAHEAbgBhAHIALwBmAHQAbwBtAHMALwAqAGgAdAB0AHAAOgAvAC8AcwB2AHIAZQBhAGwAdABvAHIAcwAuAGMAbwBtAC8AYgBpAGwAbABpAG4AZwAvAHAAOQBvAGEALwAqAGgAdAB0AHAAcwA6AC8ALwB3AGkAdwBpAGQAdwBpAG4AYQByAC4AYwBvAG0ALwBjAHIAbwB6AGoAdQBpAC8AagBGAFgASgBuAEoAcAA3AGwARAAvACcALgAiAFMAYABQAGwASQB0ACIAKABbAGMAaABhAHIAXQA0ADIAKQA7ACQAWgBiAGgAaABiAGsAcwB3AHIAYwBjAGcAZwA9ACcASwB0AHQAYwBiAHQAcgBvAHUAJwA7AGYAbwByAGUAYQBjAGgAKAAkAEIAYQBzAHgAcAB6AGgAYgBuAHEAIABpAG4AIAAkAEYAYgBnAHgAegBuAHQAbABhAG8AZgApAHsAdAByAHkAewAkAFoAeQBwAHAAcABtAG8AdQBkAGIAcQBsAC4AIgBEAG8AVwBuAEwAbwBgAEEAYABkAGYAaQBMAGUAIgAoACQAQgBhAHMAeABwAHoAaABiAG4AcQAsACAAJABMAGsAeQBqAG0AdwB2AHkAKQA7ACQARAB6AHgAdwBvAGcAYwBtAGEAZwA9ACcASgBsAG8AZwBhAG0AeAB3AG8AZAB4AHQAbAAnADsASQBmACAAKAAoACYAKAAnAEcAJwArACcAZQB0AC0ASQB0AGUAbQAnACkAIAAkAEwAawB5AGoAbQB3AHYAeQApAC4AIgBMAEUATgBHAGAAVABoACIAIAAtAGcAZQAgADMANwA0ADgAMAApACAAewAoAFsAdwBtAGkAYwBsAGEAcwBzAF0AJwB3AGkAbgAzADIAXwBQAHIAbwBjAGUAcwBzACcAKQAuACIAQwBSAEUAYABBAGAAVABlACIAKAAkAEwAawB5AGoAbQB3AHYAeQApADsAJABDAGkAbgBuAGoAcQBnAHMAeQA9ACcAVwBkAHUAdwBjAHQAeABrAHEAbQBuACcAOwBiAHIAZQBhAGsAOwAkAFgAYgB4AGEAYgBlAGsAcgA9ACcASgByAHMAZQBtAHMAZgB2AGgAZgB5AGIAdAAnAH0AfQBjAGEAdABjAGgAewB9AH0AJABWAHoAbwBtAGwAdABwAHMAbAA9ACcAVABmAHIAdQBrAHcAcgBnAGsAJwA='
    # brute_xor_key(string)
    find_key(
        r'UEsDBBQABgAIAAAAIQDJeulAAEAAOoBAAATAAAAW0NvbnRlbnRfVHlwZXNdLnhtbJSRzU7EIBDH&#xA;7yaAFqWqoHY0zpHqwe1Zj1AQhMW2I7EAbr7ts73e5ejGviEeb/8RuoN7tpFDMk8gG1vC4rKQBt&#xA;cB57Ld3T8WdFJQNOjMGBC33QHLTXF7U230EEuxG0nLIOd4rRXaAyVAZIiBPupAmk/mYehWN/TA9&#xA;qJuqulU2YAbMRV4yZFO30JnPMYvHHVvJAlGkuJhFS5dWpoYR29NZlI1o/vRUhwbSnYeNDT4SFeM&#xA;IdWvDcvkfMHR98JPk7wD8WpSfjYTYyiXaNkAweaQWFfnbSgTlSErvMWyjYRL7V6T3DnSlz4wgTz&#xA;f/Nbtr3BfEpXh59qvgEAAP//AwBQSwMEFAAGAAgAAAAhAJYFM1jUAAAAlwEAAAsAAABfcmVscy8u&#xA;cmVsc6SQPWsDMQyG90L/g9He8yVDKSWbIWsIYWuxtZ9kLNkJHNN/n1MoaVXsnWUXvQ8L9rtL2k2&#xA;C4pOTA42TQsGKXCcaHDwfnp7egGjxVP0MxM6uKLCvnt82B1x9qUe6ThlNZVC6mAsJb9aq2HE5LXh&#xA;jFSTniX5UkcZbPbh7Ae027Z9tvKbAd2KaQ7RgRziFszpmqv5DztNQVi5L03gZLnvp3CPaiN/0hGX&#xA;SvEyYHEQRbWgktTy4G979380xuYCENhaiOlfwnqfbvBnb1zu4GAAD//wMAUEsDBBQABgAIAAAA&#xA;IQAzLwWeQQAAADkAAAAUAAAAZHJzL2Nvbm5lY3RvcnhtbC54bWyysa/IzVEoSy0qzszPs1Uy1DNQ&#xA;UkjNS85PycxLt1UKDXHTtVBSKC5JzEtJzMnPS7VVqkwtVrK34UCAAAA//8DAFBLAwQUAAYACAAA&#xA;ACEAvTcvh8MAAADaAAAADwAAAGRycy9kb3ducmV2LnhtbESP3YrCMBSE7xd8h3CEvVtTi8hSjSKC&#xA;IKLggN6d2iOTbE5qU3U7tubhQUvh5n5hhlPW1uJBzWdKyg30tAEOdOl1woOOwXX98gfEDWWDkm&#xA;Bb/kYTrpfIwx07JP/TYhUJECPsMFZgQ6kxKnxuy6HuuJo7exTUWQ5RNIXWDzwi3lUyTZCgtlhwX&#xA;DNY0N5Rfd3er4FxezM2nq3QdDsfNdnDqDzfHhVKf3XY2AhGoDe/wf3upFQzg70q8AXLyAgAA//8D&#xA;AFBLAQItABQABgAIAAAAIQDJeulAAEAAOoBAAATAAAAAAAAAAAAAAAAAAAAAABbQ29udGVudF9U&#xA;eXBlc10ueG1sUEsBAi0AFAAGAAgAAAAhAJYFM1jUAAAAlwEAAAsAAAAAAAAAAAAAAAAAMQEAAF9y&#xA;ZWxzLy5yZWxzUEsBAi0AFAAGAAgAAAAhADMvBZ5BAAAAOQAAABQAAAAAAAAAAAAAAAAALgIAAGRy&#xA;cy9jb25uZWN0b3J4bWwueG1sUEsBAi0AFAAGAAgAAAAhAL03L4fDAAAA2gAAAA8AAAAAAAAAAAAA&#xA;AAAAoQIAAGRycy9kb3ducmV2LnhtbFBLBQYAAAAABAAEAPkAAACRAwAAAAA=&#xA')


if __name__ == '__main__':
    main()
