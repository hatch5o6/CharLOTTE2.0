SORTED_TEST_PAIRS = [
    (15, "sandwich", "sandwitch", 0.03), # 0
    (14, "meat", "meet", 0.24), # 2
    (13, "xxxy", "yyyy", 0.22), # 2
    (12, "stop", "pop", 0.3), # 3

    (10, "musical", "music", 0.16), # 1
    (10, "cccd", "cccc", 0.25), # 2

    (9, "lion", "leon", 0.25), # 2
    (9, "aabb", "aaaa", 0.4), # 3

    (8, "snake", "serpiente", 0.4), # 3
    (8, "giraffe", "jirafa", 0.03), # 0
    (8, "elephant", "elefante", 0.2), # 2

    (7, "orange", "naranja", 0.4), # 3
    (7, "happee", "happy", 0.2), # 2
    (7, "fridge", "freza", 0.16), # 1

    (6, "legion", "lechion", 0.05), # 0
    (6, "faint", "fanta", 0.05), # 0

    (5, "zebra", "zebra", 0.3), # 3
    (5, "ten", "dies", 0.4), # 3
    (5, "party", "fiesta", 0.4), # 3
    (5, "hello", "heko", 0.2), # 2
    
    (4, "miracle", "milagro", 0.33), # 3
    (4, "mermaid", "merman", 0.4), # 3
    (4, "mercantile", "domicile", 0.12), # 1
    (4, "horse", "herse", 0.04), # 0
    (4, "animals", "animales", 0.13), # 1

    (3, "stomach", "estomago", 0.28), # 2
    (3, "pizza", "pisa?", 0.17), # 1
    (3, "mucho", "much", 0.2), # 2

    (2, "ocean", "oceano", 0.17), # 1
    (2, "cookies", "biscuits", 0.4), # 3

    (1, "smoke", "stove", 0.3), # 3
    (1, "hungry", "hippo", 0.07), # 0
]




TEST_PAIRS = [
    (5, "hello", "heko", 0.2), # 2
    (3, "pizza", "pisa?", 0.17), # 1
    (5, "zebra", "zebra", 0.3), # 3
    (10, "musical", "music", 0.16), # 1
    (1, "hungry", "hippo", 0.07), # 0
    (6, "legion", "lechion", 0.05), # 0
    (7, "happee", "happy", 0.2), # 2
    (9, "aabb", "aaaa", 0.4), # 3
    (10, "cccd", "cccc", 0.25), # 2
    (13, "xxxy", "yyyy", 0.22), # 2
    (7, "orange", "naranja", 0.4), # 3
    (4, "miracle", "milagro", 0.33), # 3
    (2, "ocean", "oceano", 0.17), # 1
    (8, "elephant", "elefante", 0.2), # 2
    (9, "lion", "leon", 0.25), # 2
    (4, "mercantile", "domicile", 0.12), # 1
    (4, "animals", "animales", 0.13), # 1
    (3, "mucho", "much", 0.2), # 2
    (4, "horse", "herse", 0.04), # 0
    (5, "party", "fiesta", 0.4), # 3
    (7, "fridge", "freza", 0.16), # 1
    (8, "giraffe", "jirafa", 0.03), # 0
    (1, "smoke", "stove", 0.3), # 3
    (2, "cookies", "biscuits", 0.4), # 3
    (4, "mermaid", "merman", 0.4), # 3
    (6, "faint", "fanta", 0.05), # 0
    (8, "snake", "serpiente", 0.4), # 3
    (5, "ten", "dies", 0.4), # 3
    (3, "stomach", "estomago", 0.28), # 2
    (15, "sandwich", "sandwitch", 0.03), # 0
    (14, "meat", "meet", 0.24), # 2
    (12, "stop", "pop", 0.3), # 3
]

# if SORTED_TEST_PAIRS == sorted(TEST_PAIRS, reverse=True):
#     print("They are the same")
# else:
#     print("different!!!!")





TEST_PAIRS_CAP = [
    (15, 'sandwich', 'sandwitch', 0.03), # 0
    (14, 'meat', 'meet', 0.24), # 2
    (13, 'xxxy', 'yyyy', 0.22), # 2
    (12, 'stop', 'pop', 0.31), # 3

    (10, 'musical', 'music', 0.16), # 1
    (10, 'cccd', 'cccc', 0.25), # 2

    (9, 'lion', 'leon', 0.25), # 2
    (9, 'aabb', 'aaaa', 0.4), # 3

    (8, 'snake', 'serpiente', 0.4), # 3
    (8, 'giraffe', 'jirafa', 0.26), # 2
    (8, 'elephant', 'elefante', 0.21), # 2

    (7, 'orange', 'naranja', 0.4), # 3
    (7, 'happee', 'happy', 0.21), # 2
    (7, 'fridge', 'freza', 0.16), # 1

    (6, 'legion', 'lechion', 0.25), # 2
    (6, 'faint', 'fanta', 0.26), # 2

    (5, 'zebra', 'zebra', 0.31), # 3
    (5, 'ten', 'dies', 0.4), # 3
    (5, 'party', 'fiesta', 0.4), # 3
    (5, 'hello', 'heko', 0.21), # 2

    (4, 'miracle', 'milagro', 0.23), # 2
    (4, 'mermaid', 'merman', 0.4), # 3
    (4, 'mercantile', 'domicile', 0.12), # 1
    (4, 'horse', 'herse', 0.04), # 0
    (4, 'animals', 'animales', 0.23), # 2

    (3, 'stomach', 'estomago', 0.28), # 2
    (3, 'pizza', 'pisa?', 0.17), # 1
    (3, 'mucho', 'much', 0.21), # 2

    (2, 'ocean', 'oceano', 0.17), # 1
    (2, 'cookies', 'biscuits', 0.4), # 3

    (1, 'smoke', 'stove', 0.21), # 2
    (1, 'hungry', 'hippo', 0.07) # 0
]