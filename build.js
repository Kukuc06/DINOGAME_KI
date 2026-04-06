const fs = require('fs');
const path = require('path');

const files = [
  'src/NeuralNetwork.js',
  'src/GeneticAlgorithm.js',
  'src/DinoBot.js',
];

const out = files
  .map(f => fs.readFileSync(path.join(__dirname, f), 'utf8'))
  .join('\n');

fs.writeFileSync(path.join(__dirname, 'dino_bot.js'), out, 'utf8');
console.log('Built dino_bot.js');
