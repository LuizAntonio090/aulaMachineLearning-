
        let modelo;
        let modeloTreinado = false;

        async function treinarEPrever() {

            // Pegando elementos da tela
            const textoStatus = document.getElementById("status");
            const textoResultado = document.getElementById("resultado");

            // Pegando valor digitado pelo usuário
            const quantidadeDigitada = Number(document.getElementById("horas").value);

            textoStatus.innerText = "Status: Treinando a IA...";

            // =========================
            // 1. CRIAR O MODELO (neurônio)
            // =========================
            if (!modelo) {
                modelo = tf.sequential();
                modelo.add(tf.layers.dense({
                    units: 1,
                    inputShape: [1],
                    kernelInitializer: tf.initializers.randomUniform({seed: 42}),
                    biasInitializer: tf.initializers.randomUniform({seed: 42})
                }));

                // =========================
                // 2. COMPILAR O MODELO
                // =========================
                modelo.compile({
                    loss: 'meanSquaredError',
                    optimizer: 'sgd'
                });
            }

            // =========================
            // 3. DADOS DE TREINO
            // X = Quantidade de produtos 
            // Y = Preço 
            // =========================
            const dadosEntrada = tf.tensor2d([1, 2, 3, 4], [4, 1]);
            const dadosSaida = tf.tensor2d([2, 4, 6, 8], [4, 1]);

            // =========================
            // 4. TREINAMENTO
            // A IA aprende com os dados
            // =========================
            if (!modeloTreinado) {
                await modelo.fit(dadosEntrada, dadosSaida, {
                    epochs: 200,
                    shuffle: false
                });
                modeloTreinado = true;
                textoStatus.innerText = "Status: IA treinada!";
            } else {
                textoStatus.innerText = "Status: IA ja treinada!";
            }

            // =========================
            // 5. PREVISÃO
            // =========================
            const previsao = modelo.predict(
                tf.tensor2d([quantidadeDigitada], [1, 1])
            );

            // Convertendo resultado para número
            const valorPrevisto = previsao.dataSync()[0];

            // Mostrando resultado na tela
            textoResultado.innerText =
                "Preco final: " + valorPrevisto.toFixed(2);
        }
    