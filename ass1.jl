using DataFrames, CSV, Plots, Statistics, StatsBase

# data1 = CSV.read("../AIG/Assignment/student-mat.csv")
data1 = CSV.read("../Assignment/bank-additional-full.csv")
data = convert(Matrix, data1)

size(data)[2]
for i = 1:(size(data)[2])
    uniquee = unique(data[:, i])
    for j = 1:size(data)[1]
        for k = 1:size(uniquee)[1]
            if data[j,i] == uniquee[k]
                data[j,i] = Int(k)
            end
        end
    end
end




function cleanData(dataset)
    # get the grades and create the pass array
    pass = Array{Int64}(undef, size(dataset)[1], 1)

    # loop for cleaning the data
    for i in 1:size(dataset)[1]

        # 2 Job
        if (dataset[i,2] == "housemaid")
            dataset[i,2] = 1
        elseif (dataset[i,2] == "services")
            dataset[i,2] = 2
        elseif (dataset[i,2] == "technician")
            dataset[i,2] = 3
        elseif (dataset[i,2] == "blue-collar")
            dataset[i,2] = 4
        elseif (dataset[i,2] == "management")
            dataset[i,2] = 5
        elseif (dataset[i,2] == "unemployed")
            dataset[i,2] = 6
        elseif (dataset[i,2] == "retired")
            dataset[i,2] = 7
        elseif (dataset[i,2] == "admin")
            dataset[i,2] = 8
        elseif (dataset[i,2] == "technician")
            dataset[i,2] = 9
        elseif (dataset[i,2] == "entrepreneur")
            dataset[i,2] = 10
        elseif (dataset[i,2] == "unknown")
            dataset[i,2] = 11
        end

        # 15 poutcome
        if (dataset[i,15] == "nonexistent")
            dataset[i,15] = 1
        else
            dataset[i,15] = 0

        # 9 Month
        if (dataset[i,9] == "jan")
            dataset[i,9] = 1
        elseif (dataset[i,9] == "feb")
            dataset[i,9] = 2
        elseif (dataset[i,9] == "mar")
            dataset[i,9] = 3
        elseif (dataset[i,9] == "apr")
            dataset[i,9] = 4
        elseif (dataset[i,9] == "may")
            dataset[i,9] = 5
        elseif (dataset[i,9] == "jun")
            dataset[i,9] = 6
        elseif (dataset[i,9] == "jul")
            dataset[i,9] = 7
        elseif (dataset[i,9] == "aug")
            dataset[i,9] = 8
        elseif (dataset[i,9] == "sep")
            dataset[i,9] = 9
        elseif (dataset[i,9] == "oct")
            dataset[i,9] = 10
        elseif (dataset[i,9] == "nov")
            dataset[i,9] = 11
        elseif (dataset[i,9] == "dec")
            dataset[i,9] = 12
        end

        # 10 day of the week
        if (dataset[i,10] == "mon")
            dataset[i,10] = 1
        elseif (dataset[i,10] == "tue")
            dataset[i,10] = 2
        elseif (dataset[i,10] == "wed")
            dataset[i,10] = 3
        elseif (dataset[i,10] == "thu")
            dataset[i,10] = 4
        elseif (dataset[i,10] == "fri")
            dataset[i,10] = 5
        end

        # 7 loan (binary: 'yes' or 'no')
        if (dataset[i,7] == "yes")
            dataset[i,7] = 1
        elseif (dataset[i,7] == "no")
            dataset[i,7] = 0
        end

        # 21 y (binary: yes or no)
        if (dataset[i,21] == "yes")
            dataset[i,21] = 1
        else
            dataset[i,21] = 0
        end

            # 3 Marital Status (nominal: person is 'married', 'single' or 'divorced')
            if (dataset[i,3] == "married")
                dataset[i,3] = 1
            elseif (dataset[i,3] == "single")
                dataset[i,3] = 2
            elseif (dataset[i,3] == "divorced")
                dataset[i,11] = 3
            end

            # get the duration mean
            if (mean(duration[i,:]) >= 10)
                pass[i] = 1
            else
                pass[i] = 0
            end

            # 5 default (binary: yes or no)
            if (dataset[i,5] == "yes")
                dataset[i,5] = 1
            else dataset[i,5] = 0
            end

            # 6 housing (binary: yes or no)
            if (dataset[i,6] == "yes")
            dataset[i,6] = 1
           else dataset[i,6] = 0
            end
        end
    end
end

function hypothesis(x, theeta)
    z = theeta' * x
    return 1 / (1 + exp(z))
end

function gradientDescent(x, y, theta, m, n, rp, lr)
    c = 0
    s = 0
    for i in 1:m
        c += -y[i] * log(hypothesis(x[i,:], theta)) - (1 - y[i]) * log(1 - hypothesis(x[i,:], theta))
        for j in 2:n
            s += (theta[j])^2
        end
        s *= rp / (2 * m)
        c *= 1 / m
        c += s
        theta = theta - lr * (1 / m * x[i,:] * (hypothesis(x[i,:], theta) - y[i]) + (rp / m) * theta)
        println("cost: ", c)
    end

    return theta
end

function standardization(x)
    dt = StatsBase.fit(ZScoreTransform, x, dims = 2)
    return StatsBase.transform(dt, x)
end

cd = cleanData(data)

x = cd[data]
x = standardization(x)
y = cd[data]
oneMatrix = ones(size(x)[1])
x = hcat(x, oneMatrix)
theta = zeros(size(x)[2])

i = trunc(Int, (size(x)[1]) * 0.8)

# training data
x_train = x[1:i,:]
x_test = x[i + 1:size(x)[1],:]

y_train = y[1:i,:]
y_test = y[i + 1:size(x)[1],:]


df = data[:, end - 2:end]
apu = fill(1, size(data)[1])
for i = 1:size(df)[1]
    if sum(df[i, :]) / 3.0 >= 10
        apu[i] = 1
    else apu[i] = 0
    end
end