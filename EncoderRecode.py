import time

tokensFileName = 'tokens2.txt'
startTime = time.time()
lastTime = time.time()

tokens_size = 1000000
special_tokens = tokens_size + 5

with open('Formatted.txt', 'r', encoding='ANSI') as f:
    text = f.read()

with open(tokensFileName, 'r', encoding='ANSI') as tokensFileRead:
    tokensList = tokensFileRead.readlines()

def getStats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        # Skip pairs that contain special token
        if special_tokens not in pair:
            counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

merges = {}
vocab = {idx: bytes([idx]) for idx in range(256)}
vocabLen = None

def getTime():
    global lastTime
    print('- - - - -')
    print(f'Program Runtime: {time.time() - startTime}')
    print(f'Time since last call: {time.time() - lastTime}')
    print('- - - - -')
    lastTime = time.time()

def getTokensFromFile():
    global merges
    global vocab
    global vocabLen
    
    for line in tokensList:
        if line:
            nums = list(map(int, line.split()))
            merges[(nums[0], nums[1])] = nums[2]
            #print(nums, nums[0], nums[1], nums[2])
    print(f'Merges length: {len(merges)}')
    getTime()
    #print(merges.items())
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    vocabLen = len(vocab)
    print(f'Vocabulary length: {vocabLen}')
    getTime()

def getNewTokens(vocabSize):
    global merges
    global vocab
    global vocabLen
    global tokens

    tokens = text.encode("ANSI")
    tokens = list(map(int, tokens))
    eof = '<endf>'.encode("ANSI")
    eof = list(map(int, eof))

    def mergeTokens(ids, num_merges, special_token, additional_ids):
        global merges
        global vocab
        global vocabLen
        global tokens
        start = special_token - num_merges + 1 if special_token else 256
        for idx in range(start, start + num_merges):
            if additional_ids:
                stats = getStats(additional_ids)
                print(stats)
            else:
                stats = getStats(ids)
                
            if not stats:  # If no valid pairs left (all contain special token <endf>)
                print('No valid tokens left')
                break
            pair = max(stats, key=stats.get)
            print(f'Merging {str(pair)} into a new token: {str(idx)}')
            getTime()
            ids = merge(ids, pair, idx)
            if additional_ids:
                additional_ids = merge(additional_ids, pair, idx)
            merges[pair] = idx
            with open(tokensFileName, 'a') as tokensFileWrite:
                tokensFileWrite.write(' '.join(map(str, pair)) + ' ' + str(merges[pair]) + '\n')
        tokens = ids

    merges = {}
    mergeTokens(list(tokens), len(list(eof))-1, tokens_size + len(list(eof)) - 1, list(eof))
    input()
    mergeTokens(list(tokens), vocabSize - 256, 0, [])
    
    print(f'Merges length: {len(merges)}')
    getTime()

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    vocabLen = len(vocab)
    print(f'Vocabulary length: {vocabLen}')
    getTime()

if tokensList:
    getTokensFromFile()
else:
    getNewTokens(tokens_size)

def encode(text):
    tokens = list(text.encode("ANSI"))
    result = []

    for token in tokens:
        result.append(token)
        while len(result) >= 2:
            pair = (result[-2], result[-1])
            if pair in merges:
                result[-2:] = [merges[pair]]
            else:
                break
            
    return result

def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("ANSI")
    
    return text

tokenizedText = encode(text)
print("length:", len(text))
print("- - - - -")
print("tokens length:", len(tokenizedText))
print("compression ratio:", str(round(len(text)/len(tokenizedText),2)) + "x")
getTime()
print(f'First fifteen tokens: {tokenizedText[:16]}...')
print(f'First fifteen tokens decoded: {decode(tokenizedText[:16])}...')
print(decode([0]))
while True:
    message = input().lower()
    tokenizedMessage = encode(message)
    print(tokenizedMessage)
    print(*[decode([i]) for i in tokenizedMessage], sep = '/')
    #print(decode(tokenizedMessage))
    print('- - - - -')
