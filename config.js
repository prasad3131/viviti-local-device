const path = require('path');
module.exports = {
  port: process.env.PORT || 3000,
  photoDir: process.env.PHOTO_DIR || '/home/orangepi/photos',
  dataDir: process.env.DATA_DIR || path.join(__dirname, 'data'),
};
