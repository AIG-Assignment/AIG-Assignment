using DataFrames, CSV, Plots

#data1 = CSV.read("C:/Users/PUBLIC.DESKTOP-2E4MMQ1/Desktop/AIG/bank-marketing-campaign/bank-additional-full.csv")
data1 = CSV.read("C:/Users/PUBLIC.DESKTOP-2E4MMQ1/Desktop/AIG/bank-marketing-campaign/bank-additional-full.csv")
data = convert(Matrix, data1)

size(data)[2]
for i = 1:(size(data)[2])
    same = unique(data[:, i])
    for j = 1:size(data)[1]
        for k = 1:size(same)[1]
            if data[j,i] == same[k]
                data[j,i] = Int(k)
            end
        end
    end
end

tricks = data[:, end-2: end]
ari = fill(1, size(data)[1])
for i = 1:size(tricks)[1]
    if sum(tricks[i, :])/3.0 >= 10
        ari[i] = 1
    else ari[i] = 0
    end
    println()
end

testst = data[:,end-2]+data[:,end-1]+data[:,end]
testst./3
equates = data[:,1:end-2]
values = ari
indd = ceil(0.8*size(equates,1))
splitt = convert(Int,indd)
train_data = equates[1:splitt,:]
train_values = values[1:splitt]
bias = fill(1, splitt)
train_data = hcat(train_data,bias)
train_data = train_data./maximum(train_data)

theta = ari = fill(0.0, size(train_data)[2])
steps = 1000
alpha = 1
m = size(train_data,1)
 lambda = 1
for i = 1:steps
       ari = transpose(theta)*(transpose(train_data) .- transpose(train_values))
       global theta = ((theta) .- alpha/m .* (transpose(train_data)*transpose(ari)
       .+ lambda.*(theta)))
       end
print(theta)
print(error)
1 .- theta